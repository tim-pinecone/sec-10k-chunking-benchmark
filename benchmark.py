"""
Multi-config chunking benchmark against the SEC 10-K QA dataset.

Execution is two-phase:
  Phase 1 — Embed upfront (zero API calls during evaluation):
    • Questions embedded once per model, streamed to parquet on S3
    • For each config: corpus chunked, chunks embedded in batches,
      streamed to parquet on S3 — never accumulates all vectors in memory

  Phase 2 — Evaluate (all cache hits):
    • Parquet loaded back via pyarrow fixed-size list → numpy (zero Python float overhead)
    • MTCB SimpleEvaluator runs with pre-populated in-memory cache
    • Results checkpointed to S3 after each config

Usage:
    python benchmark.py
    python benchmark.py --configs RecursiveChunker:512,TokenChunker:256
    python benchmark.py --embedding-model openai:text-embedding-3-large
    python benchmark.py --upload-hf Tim-Pinecone/sec-10k-qa-embeddings
"""

import argparse
import gc
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config matrix
# ---------------------------------------------------------------------------
DEFAULT_CONFIGS = [
    ("RecursiveChunker", 256),
    ("RecursiveChunker", 512),
    ("RecursiveChunker", 1024),
    ("SentenceChunker",  256),
    ("SentenceChunker",  512),
    ("SentenceChunker",  1024),
    ("TokenChunker",     256),
    ("TokenChunker",     512),
    ("TokenChunker",     1024),
]

K_VALUES  = [1, 3, 5, 10]
S3_BUCKET = os.environ.get("S3_BUCKET", "mtcb-benchmark")
S3_PREFIX = "sec-benchmark"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--configs", default=None)
parser.add_argument(
    "--embedding-model",
    default="BAAI/bge-large-en-v1.5",
    help="HuggingFace model ID or openai:<model-name>",
)
parser.add_argument("--upload-hf", default=None, metavar="DATASET_ID")
parser.add_argument(
    "--embed-batch-size",
    type=int,
    default=512,
    help="Texts per embedding API call and parquet row-group (default: 512)",
)
parser.add_argument(
    "--run-name",
    default="",
    help="Namespace for S3 paths and local output, e.g. 'v2'. Empty = root.",
)
args = parser.parse_args()

if args.configs:
    CONFIGS = []
    for item in args.configs.split(","):
        name, size = item.split(":")
        # chunk_size=0 means the chunker doesn't use a fixed size (e.g. NeuralChunker)
        CONFIGS.append((name.strip(), int(size.strip())))
else:
    CONFIGS = DEFAULT_CONFIGS

EMBEDDING_MODEL_ARG = args.embedding_model
EMBED_BATCH_SIZE    = args.embed_batch_size
RUN_NAME            = args.run_name.strip("/")

# S3 prefix and local dirs are namespaced by --run-name
_s3_run = f"sec-benchmark/{RUN_NAME}" if RUN_NAME else "sec-benchmark"
S3_PREFIX = _s3_run

RESULTS_DIR    = Path("results") / RUN_NAME if RUN_NAME else Path("results")
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
CACHE_DIR      = Path("cache")
DATASET_DIR    = Path("dataset")
for d in (RESULTS_DIR, EMBEDDINGS_DIR, CACHE_DIR, DATASET_DIR):
    d.mkdir(exist_ok=True, parents=True)

timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_PATH   = RESULTS_DIR / f"benchmark_{timestamp}.csv"
S3_CHECKPOINT_KEY = f"{S3_PREFIX}/results/benchmark_{timestamp}.csv"

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
_s3 = boto3.client("s3") if S3_BUCKET else None


def s3_upload(local_path: Path, key: str):
    if _s3:
        try:
            _s3.upload_file(str(local_path), S3_BUCKET, key)
            print(f"    ☁️  → s3://{S3_BUCKET}/{key}")
        except Exception as e:
            print(f"    ⚠️  S3 upload failed ({key}): {e}")


def s3_download(key: str, local_path: Path) -> bool:
    if _s3:
        try:
            _s3.download_file(S3_BUCKET, key, str(local_path))
            return True
        except Exception:
            return False
    return False


def s3_exists(key: str) -> bool:
    if not _s3:
        return False
    try:
        _s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
def _model_key(model_arg: str) -> str:
    return model_arg.removeprefix("openai:").split("/")[-1].replace(".", "-")


MODEL_KEY = _model_key(EMBEDDING_MODEL_ARG)

print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL_ARG}  (key: {MODEL_KEY})")

if EMBEDDING_MODEL_ARG.startswith("openai:"):
    from chonkie import OpenAIEmbeddings
    _base_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_ARG.removeprefix("openai:"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
else:
    from chonkie import SentenceTransformerEmbeddings
    _base_model = SentenceTransformerEmbeddings(EMBEDDING_MODEL_ARG)


class _CachedEmbeddings:
    """Wraps any chonkie embedding with an in-memory dict cache.
    MTCB passes extra kwargs (input_type, show_progress) — we drop them here.
    """
    def __init__(self, base):
        self._base = base
        self._cache: dict = {}

    def embed(self, text, **_):
        if text not in self._cache:
            self._cache[text] = np.asarray(self._base.embed(text), dtype=np.float32)
        return self._cache[text]

    def embed_batch(self, texts, **_):
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            vecs = self._base.embed_batch(uncached)
            for t, v in zip(uncached, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float32)
        return [self._cache[t] for t in texts]

    def __getattr__(self, name):
        return getattr(self._base, name)


embeddings = _CachedEmbeddings(_base_model)

# ---------------------------------------------------------------------------
# Streaming parquet helpers
# ---------------------------------------------------------------------------
_EMBED_DIM: int | None = None


def _get_dim() -> int:
    global _EMBED_DIM
    if _EMBED_DIM is None:
        _EMBED_DIM = len(_base_model.embed("dim_probe"))
    return _EMBED_DIM


def _emb_col_from_matrix(matrix: np.ndarray) -> pa.Array:
    """float32 (n, dim) numpy → pyarrow FixedSizeListArray, zero-copy."""
    dim  = matrix.shape[1]
    flat = pa.array(matrix.flatten(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def _embed_batch_to_cache(texts: list):
    """Embed any texts not already in cache, store as float32 numpy."""
    uncached = [t for t in texts if t not in embeddings._cache]
    if uncached:
        vecs = _base_model.embed_batch(uncached)
        for t, v in zip(uncached, vecs):
            embeddings._cache[t] = np.asarray(v, dtype=np.float32)


def embed_questions_to_parquet(questions: list, path: Path, batch_size: int) -> Path:
    """Stream question embeddings to parquet.

    Schema: question_id (int32), question (large_string), embedding (list<float32, dim>)
    """
    dim    = _get_dim()
    schema = pa.schema([
        ("question_id", pa.int32()),
        ("question",    pa.large_string()),
        ("embedding",   pa.list_(pa.float32(), dim)),
    ])
    writer  = pq.ParquetWriter(str(path), schema, compression="snappy")
    total   = len(questions)
    written = 0
    for start in range(0, total, batch_size):
        batch_q   = questions[start : start + batch_size]
        batch_ids = list(range(start, start + len(batch_q)))
        _embed_batch_to_cache(batch_q)
        matrix = np.stack([embeddings._cache[q] for q in batch_q])
        writer.write_table(pa.table({
            "question_id": pa.array(batch_ids, type=pa.int32()),
            "question":    pa.array(batch_q,   type=pa.large_string()),
            "embedding":   _emb_col_from_matrix(matrix),
        }, schema=schema))
        written += len(batch_q)
        print(f"    {written:,}/{total:,}", end="\r")
    writer.close()
    print(f"    {written:,}/{total:,} questions embedded ✓")
    return path


def embed_chunks_to_parquet(corpus: list, chunker, path: Path, batch_size: int) -> Path:
    """Chunk corpus and stream embeddings to parquet.

    Schema: doc_id (int32), chunk_idx (int32), chunk_text (large_string),
            embedding (list<float32, dim>)

    Processes one document at a time so peak RAM = one doc's chunks.
    """
    dim    = _get_dim()
    schema = pa.schema([
        ("doc_id",     pa.int32()),
        ("chunk_idx",  pa.int32()),
        ("chunk_text", pa.large_string()),
        ("embedding",  pa.list_(pa.float32(), dim)),
    ])
    writer       = pq.ParquetWriter(str(path), schema, compression="snappy")
    total_chunks = 0

    # Accumulate across docs until we have a full batch, then flush
    buf_doc_ids  = []
    buf_chunk_ids = []
    buf_texts     = []

    def _flush(force=False):
        nonlocal buf_doc_ids, buf_chunk_ids, buf_texts
        if not buf_texts or (not force and len(buf_texts) < batch_size):
            return
        _embed_batch_to_cache(buf_texts)
        matrix = np.stack([embeddings._cache[t] for t in buf_texts])
        writer.write_table(pa.table({
            "doc_id":     pa.array(buf_doc_ids,   type=pa.int32()),
            "chunk_idx":  pa.array(buf_chunk_ids, type=pa.int32()),
            "chunk_text": pa.array(buf_texts,     type=pa.large_string()),
            "embedding":  _emb_col_from_matrix(matrix),
        }, schema=schema))
        buf_doc_ids, buf_chunk_ids, buf_texts = [], [], []

    for doc_id, doc in enumerate(corpus):
        for chunk_idx, chunk in enumerate(chunker(doc)):
            buf_doc_ids.append(doc_id)
            buf_chunk_ids.append(chunk_idx)
            buf_texts.append(chunk.text)
            total_chunks += 1
            if len(buf_texts) >= batch_size:
                _flush()
                print(f"    {total_chunks:,} chunks embedded", end="\r")

    _flush(force=True)
    writer.close()
    print(f"    {total_chunks:,} chunks embedded ✓")
    return path


def load_parquet_to_cache(path: Path):
    """Load any embedding parquet into embeddings._cache.

    Reads the embedding column as a flat numpy buffer — no Python float objects.
    Handles question parquet (text col = 'question') and chunk parquet ('chunk_text').
    """
    table = pq.read_table(str(path))
    n     = len(table)
    if n == 0:
        return

    names    = table.schema.names
    text_col = next((c for c in ["chunk_text", "question", "text"] if c in names), None)
    if text_col is None:
        raise ValueError(f"No text column found in {path.name}: {names}")

    texts    = table.column(text_col).to_pylist()
    flat_arr = table.column("embedding").combine_chunks().flatten()
    flat_np  = flat_arr.to_numpy(zero_copy_only=False).astype(np.float32)
    matrix   = flat_np.reshape(n, len(flat_np) // n)

    for i, text in enumerate(texts):
        embeddings._cache[text] = matrix[i]

    print(f"    Loaded {n:,} embeddings from {path.name}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print("\n📥 Loading dataset...")
corpus_local    = DATASET_DIR / "corpus.parquet"
questions_local = DATASET_DIR / "questions.parquet"

if not corpus_local.exists() or not questions_local.exists():
    corpus_ok    = s3_download(f"{S3_PREFIX}/dataset/corpus.parquet",    corpus_local)
    questions_ok = s3_download(f"{S3_PREFIX}/dataset/questions.parquet", questions_local)
    if corpus_ok and questions_ok:
        print("  Loaded from S3.")
    else:
        print("  S3 miss — loading from HuggingFace...")
        from datasets import load_dataset as _load
        _ds_c = _load("Tim-Pinecone/sec-10k-qa", "corpus",    split="train")
        _ds_q = _load("Tim-Pinecone/sec-10k-qa", "questions", split="train")
        _ds_c.to_parquet(str(corpus_local))
        _ds_q.to_parquet(str(questions_local))
        s3_upload(corpus_local,    f"{S3_PREFIX}/dataset/corpus.parquet")
        s3_upload(questions_local, f"{S3_PREFIX}/dataset/questions.parquet")
        print("  Loaded from HuggingFace and cached to S3.")

from datasets import load_dataset
ds_corpus    = load_dataset("parquet", data_files=str(corpus_local),    split="train")
ds_questions = load_dataset("parquet", data_files=str(questions_local), split="train")

corpus            = [row["text"]               for row in ds_corpus]
questions         = [row["question"]           for row in ds_questions]
relevant_passages = [row["chunk_must_contain"] for row in ds_questions]

print(f"  {len(corpus)} documents, {len(questions)} QA pairs")
if not questions:
    raise SystemExit("❌ No questions loaded.")

# ---------------------------------------------------------------------------
# Chunker factory
# ---------------------------------------------------------------------------
from chonkie import (
    RecursiveChunker, SentenceChunker, TokenChunker,
    SemanticChunker, LateChunker, NeuralChunker,
)

CHUNKER_MAP = {
    "RecursiveChunker": RecursiveChunker,
    "SentenceChunker":  SentenceChunker,
    "TokenChunker":     TokenChunker,
    "SemanticChunker":  SemanticChunker,
    "LateChunker":      LateChunker,
    "NeuralChunker":    NeuralChunker,
}

# ---------------------------------------------------------------------------
# Result parsing — attribute access first, regex fallback
# ---------------------------------------------------------------------------
def _parse_recall(result, k):
    for attr in [f"recall_at_{k}", f"r_at_{k}"]:
        v = getattr(result, attr, None)
        if v is not None:
            return float(v)
    recall = getattr(result, "recall", None)
    if isinstance(recall, dict) and k in recall:
        return float(recall[k])
    m = re.search(rf"R@{k}=([\d.]+)%", str(result))
    return float(m.group(1)) / 100 if m else None


def _parse_mrr(result, k):
    mrr = getattr(result, "mrr", None)
    if isinstance(mrr, dict) and k in mrr:
        return float(mrr[k])
    m = re.search(rf"MRR@{k}=([\d.]+)", str(result))
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------
from mtcb import SimpleEvaluator

completed = set()
results   = []

try:
    objs = _s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/results/").get("Contents", [])
    csvs = sorted([o["Key"] for o in objs if o["Key"].endswith(".csv")], reverse=True)
    if csvs:
        local_resume = RESULTS_DIR / "resume.csv"
        if s3_download(csvs[0], local_resume):
            df_prev = pd.read_csv(local_resume)
            results = df_prev.to_dict("records")
            for row in df_prev.itertuples():
                if not (hasattr(row, "error") and pd.notna(getattr(row, "error", None))):
                    completed.add((
                        str(row.chunker),
                        int(row.chunk_size),
                        str(getattr(row, "embedding_model", "")),
                    ))
            print(f"\n📂 Resuming: {len(completed)} configs already done (from {csvs[0]})")
            CHECKPOINT_PATH   = local_resume
            S3_CHECKPOINT_KEY = csvs[0]
except Exception as e:
    print(f"  (no checkpoint: {e})")

# ---------------------------------------------------------------------------
# Phase 1 — Embed questions
# ---------------------------------------------------------------------------
q_parquet_name = f"questions_{MODEL_KEY}.parquet"
q_parquet_path = EMBEDDINGS_DIR / q_parquet_name
q_s3_key       = f"{S3_PREFIX}/embeddings/{q_parquet_name}"

if s3_exists(q_s3_key):
    print(f"\n✅ Question embeddings already on S3: {q_parquet_name}")
    if not q_parquet_path.exists():
        s3_download(q_s3_key, q_parquet_path)
    load_parquet_to_cache(q_parquet_path)
else:
    print(f"\n🔢 Phase 1 — Embedding {len(questions)} questions...")
    embed_questions_to_parquet(questions, q_parquet_path, batch_size=EMBED_BATCH_SIZE)
    s3_upload(q_parquet_path, q_s3_key)

# ---------------------------------------------------------------------------
# Phase 2 — Per-config: chunk, embed, evaluate
# ---------------------------------------------------------------------------
total = len(CONFIGS)
print(f"\n{'='*60}")
print(f"Phase 2 — {total} configs  |  model: {EMBEDDING_MODEL_ARG}")
print(f"{'='*60}")

for i, (chunker_name, chunk_size) in enumerate(CONFIGS, 1):
    run_key = (chunker_name, chunk_size, EMBEDDING_MODEL_ARG)
    if run_key in completed:
        print(f"\n[{i}/{total}] ⏭️  {chunker_name}(size={chunk_size}) — already done")
        continue

    print(f"\n[{i}/{total}] {chunker_name}(size={chunk_size})")
    start = time.time()
    row = {
        "chunker":         chunker_name,
        "chunk_size":      chunk_size,
        "embedding_model": EMBEDDING_MODEL_ARG,
    }

    try:
        cls     = CHUNKER_MAP.get(chunker_name)
        if cls is None:
            raise ValueError(f"Unknown chunker: {chunker_name}")
        # NeuralChunker determines boundaries automatically — no chunk_size
        chunker = cls() if chunk_size == 0 else cls(chunk_size=chunk_size)

        chunk_parquet_name = f"chunks_{chunker_name}_{chunk_size}_{MODEL_KEY}.parquet"
        chunk_parquet_path = EMBEDDINGS_DIR / chunk_parquet_name
        chunk_s3_key       = f"{S3_PREFIX}/embeddings/{chunk_parquet_name}"

        if s3_exists(chunk_s3_key):
            print(f"  ✅ Chunk embeddings already on S3: {chunk_parquet_name}")
            if not chunk_parquet_path.exists():
                s3_download(chunk_s3_key, chunk_parquet_path)
            load_parquet_to_cache(chunk_parquet_path)
        else:
            print(f"  Chunking + embedding corpus...")
            embed_chunks_to_parquet(corpus, chunker, chunk_parquet_path, batch_size=EMBED_BATCH_SIZE)
            s3_upload(chunk_parquet_path, chunk_s3_key)

        # Evaluate — all embed_batch calls are cache hits
        print(f"  ▶️  Evaluating...")
        evaluator = SimpleEvaluator(
            corpus=corpus,
            questions=questions,
            relevant_passages=relevant_passages,
            chunker=chunker,
            embedding_model=embeddings,
        )
        result  = evaluator.evaluate(k=K_VALUES)
        elapsed = time.time() - start

        row["time_sec"] = round(elapsed, 1)
        for k in K_VALUES:
            row[f"R@{k}"]   = _parse_recall(result, k)
            row[f"MRR@{k}"] = _parse_mrr(result, k)

        print(f"  ✅  R@1={row.get('R@1'):.1%}  R@5={row.get('R@5'):.1%}  "
              f"R@10={row.get('R@10'):.1%}  ({elapsed:.0f}s)")

    except Exception as e:
        elapsed = time.time() - start
        row["time_sec"] = round(elapsed, 1)
        row["error"]    = str(e)
        print(f"  ❌  {e} ({elapsed:.0f}s)")

    results.append(row)
    completed.add(run_key)
    pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
    s3_upload(CHECKPOINT_PATH, S3_CHECKPOINT_KEY)
    gc.collect()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
df    = pd.DataFrame(results)
df_ok = df[~df["error"].notna()].copy() if "error" in df.columns else df.copy()

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
show_cols = ["chunker", "chunk_size"] + [f"R@{k}" for k in K_VALUES] + ["time_sec"]
show_cols = [c for c in show_cols if c in df.columns]
print(df[show_cols].to_string(index=False))

if "R@5" in df_ok.columns and len(df_ok) > 0:
    print(f"\nAvg R@5 by chunker:")
    for name, val in df_ok.groupby("chunker")["R@5"].mean().sort_values(ascending=False).items():
        print(f"  {name:25s}  {val:.1%}")

print(f"\n💾 {CHECKPOINT_PATH}")
print(f"☁️  s3://{S3_BUCKET}/{S3_CHECKPOINT_KEY}")
print(f"\n📁 Embedding parquets:")
for p in sorted(EMBEDDINGS_DIR.glob("*.parquet")):
    print(f"  {p.name}  ({p.stat().st_size / 1_048_576:.1f} MB)")

# ---------------------------------------------------------------------------
# Optional HF upload
# ---------------------------------------------------------------------------
if args.upload_hf:
    print(f"\n🤗 Uploading to {args.upload_hf}...")
    from datasets import Dataset, DatasetDict
    splits = {}
    for p in sorted(EMBEDDINGS_DIR.glob("*.parquet")):
        splits[p.stem.replace("-", "_")] = Dataset(pq.read_table(p))
    DatasetDict(splits).push_to_hub(args.upload_hf, token=os.environ.get("HF_TOKEN"))
    print(f"  ✅ Pushed {len(splits)} splits")
