"""
Re-evaluate chunking results at any k value using pre-computed embeddings from S3.

Downloads question + chunk embedding parquets from S3 and runs pure numpy
vector search — no API calls, no GPU, no re-embedding.

Usage:
    uv run python reeval.py
    uv run python reeval.py --k 1,3,5,10,20,30
    uv run python reeval.py --model text-embedding-ada-002 --k 1,5,10,20,30
    uv run python reeval.py --embeddings-prefix sec-benchmark/embeddings
"""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--k",
    default="1,3,5,10,20,30",
    help="Comma-separated k values (default: 1,3,5,10,20,30)",
)
parser.add_argument(
    "--model",
    default="text-embedding-ada-002",
    help="Model key used in parquet filenames (default: text-embedding-ada-002)",
)
parser.add_argument(
    "--embeddings-prefix",
    default="sec-benchmark/embeddings",
    help="S3 prefix where embedding parquets live",
)
parser.add_argument(
    "--s3-bucket",
    default=os.environ.get("S3_BUCKET", "mtcb-benchmark"),
)
parser.add_argument(
    "--out",
    default="results/reeval.csv",
    help="Where to write the results CSV",
)
args = parser.parse_args()

K_VALUES = [int(k.strip()) for k in args.k.split(",")]
MODEL    = args.model
PREFIX   = args.embeddings_prefix.rstrip("/")
BUCKET   = args.s3_bucket
OUT      = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_s3_parquets(prefix: str) -> list[str]:
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])
    return [o["Key"] for o in objs if o["Key"].endswith(".parquet")]


def download(key: str, local: Path) -> Path:
    s3.download_file(BUCKET, key, str(local))
    return local


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (texts, matrix) from an embedding parquet.
    Matrix shape: (n, dim), dtype float32.
    """
    table    = pq.read_table(str(path))
    names    = table.schema.names
    text_col = next((c for c in ["chunk_text", "question", "text"] if c in names), None)
    if text_col is None:
        raise ValueError(f"No text column in {path.name}: {names}")

    texts    = table.column(text_col).to_pylist()
    flat_arr = table.column("embedding").combine_chunks().flatten()
    flat_np  = flat_arr.to_numpy(zero_copy_only=False).astype(np.float32)
    matrix   = flat_np.reshape(len(texts), len(flat_np) // len(texts))
    return texts, matrix


def cosine_sim(Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return (n_q, n_c) cosine similarity matrix."""
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return Q @ C.T


def evaluate(
    q_matrix:  np.ndarray,
    c_matrix:  np.ndarray,
    chunk_texts: list[str],
    relevant_passages: list[str],
    k_values:  list[int],
) -> dict:
    """Pure numpy evaluation. Returns dict of R@k and MRR@k."""
    max_k = max(k_values)
    sim   = cosine_sim(q_matrix, c_matrix)               # (n_q, n_c)
    top_i = np.argsort(-sim, axis=1)[:, :max_k]          # (n_q, max_k)

    ranks = []
    for q_idx, relevant in enumerate(relevant_passages):
        rank = 0
        for r, chunk_idx in enumerate(top_i[q_idx], 1):
            if relevant in chunk_texts[chunk_idx]:
                rank = r
                break
        ranks.append(rank)

    out = {}
    for k in k_values:
        out[f"R@{k}"]   = sum(1 for r in ranks if 0 < r <= k) / len(ranks)
        out[f"MRR@{k}"] = float(np.mean([1 / r if 0 < r <= k else 0.0 for r in ranks]))
    return out


# ---------------------------------------------------------------------------
# Load questions dataset for relevant_passages
# ---------------------------------------------------------------------------
print(f"📥 Loading dataset...")
from datasets import load_dataset

ds_q = load_dataset("Tim-Pinecone/sec-10k-qa", "questions", split="train")
questions         = [row["question"]           for row in ds_q]
relevant_passages = [row["chunk_must_contain"] for row in ds_q]
print(f"   {len(questions)} questions loaded")

# ---------------------------------------------------------------------------
# Discover available parquets on S3
# ---------------------------------------------------------------------------
print(f"\n🔍 Scanning s3://{BUCKET}/{PREFIX}/ for model={MODEL}...")
all_keys    = list_s3_parquets(PREFIX)
q_keys      = [k for k in all_keys if f"questions_{MODEL}" in k]
chunk_keys  = [k for k in all_keys if f"chunks_" in k and MODEL in k]

if not q_keys:
    raise SystemExit(f"❌ No question parquet found for model '{MODEL}' under {PREFIX}/")

print(f"   Question parquet : {q_keys[0].split('/')[-1]}")
print(f"   Chunk parquets   : {len(chunk_keys)} configs")
for k in sorted(chunk_keys):
    print(f"     {k.split('/')[-1]}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
results = []

with TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # Load question embeddings once
    print(f"\n📦 Loading question embeddings...")
    q_path = download(q_keys[0], tmp / "questions.parquet")
    _, q_matrix = load_embeddings(q_path)
    print(f"   {q_matrix.shape[0]} questions × dim {q_matrix.shape[1]}")

    # Evaluate each config
    print(f"\n{'='*60}")
    print(f"Evaluating at k={K_VALUES}")
    print(f"{'='*60}")

    for key in sorted(chunk_keys):
        fname     = key.split("/")[-1]           # e.g. chunks_LateChunker_512_text-embedding-ada-002.parquet
        stem      = fname.replace(".parquet", "").removeprefix("chunks_").removesuffix(f"_{MODEL}")
        # stem is now e.g. LateChunker_512 or NeuralChunker_0
        parts     = stem.rsplit("_", 1)
        chunker_name = parts[0]
        chunk_size   = int(parts[1]) if len(parts) > 1 else 0

        print(f"\n{chunker_name}(size={chunk_size})")
        local = download(key, tmp / fname)
        chunk_texts, c_matrix = load_embeddings(local)
        print(f"  {c_matrix.shape[0]:,} chunks × dim {c_matrix.shape[1]}")

        metrics = evaluate(q_matrix, c_matrix, chunk_texts, relevant_passages, K_VALUES)

        row = {
            "chunker":         chunker_name,
            "chunk_size":      chunk_size,
            "embedding_model": MODEL,
            "n_chunks":        len(chunk_texts),
        }
        row.update(metrics)
        results.append(row)

        recall_str = "  ".join(f"R@{k}={metrics[f'R@{k}']:.1%}" for k in K_VALUES)
        print(f"  {recall_str}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
show = ["chunker", "chunk_size"] + [f"R@{k}" for k in K_VALUES]
show = [c for c in show if c in df.columns]
print(df[show].to_string(index=False))

if f"R@{max(K_VALUES)}" in df.columns:
    print(f"\nRanked by R@{max(K_VALUES)}:")
    ranked = df[show].sort_values(f"R@{max(K_VALUES)}", ascending=False)
    print(ranked.to_string(index=False))

df.to_csv(OUT, index=False)
print(f"\n💾 {OUT}")
