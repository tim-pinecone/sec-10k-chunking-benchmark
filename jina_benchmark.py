"""
Jina v5 chunking benchmark — custom RecursiveChunker + LateChunker.

Uses jinaai/jina-embeddings-v5-text-small loaded via transformers.
Late chunking accesses raw token hidden states and applies mean-pooling per
chunk span, following the technique from:
  https://jina.ai/news/late-chunking-in-long-context-embedding-models

NOTE: Jina v5 uses last-token pooling by design (Qwen3-0.6B base). Applying
mean-pooling over hidden state spans is a methodological experiment to test
whether the late chunking technique transfers to decoder-based embedding models.
Results should be interpreted with this caveat in mind.

Context window: 32,768 tokens. SEC 10-K filings are 50K-150K tokens.
Late chunking is applied per non-overlapping 32K-token window — cross-chunk
context is shared within a window, not across the full document.

Usage:
    python jina_benchmark.py
    python jina_benchmark.py --dry-run
    python jina_benchmark.py --batch-size 8
"""

import argparse
import gc
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
import torch
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID    = "jinaai/jina-embeddings-v5-text-small"
EMBED_DIM   = 1024
MAX_SEQ_LEN = 32768
MODEL_KEY   = "jina-v5"

# Instruction prefix for queries (retrieval task)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

S3_BUCKET = os.environ.get("S3_BUCKET", "mtcb-benchmark")
S3_PREFIX = "sec-benchmark/jina-v5"

K_VALUES = [1, 5, 10, 20, 30]

SEPARATORS = [
    "\n\n", "\n", " ", ".", ",",
    "\u200b",  # Zero-width space
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    "",
]

CONFIGS = [
    ("RecursiveChunker", 512),
    ("RecursiveChunker", 1024),
    ("LateChunker",      512),
    ("LateChunker",      1024),
]

# Separator used when concatenating chunks into a late-chunking window
_WIN_SEP = " "

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Process only first 5 docs for smoke testing")
parser.add_argument("--batch-size", type=int, default=16,
                    help="Texts per encode call for RecursiveChunker/questions (default: 16)")
args = parser.parse_args()

RESULTS_DIR    = Path("results")
EMBEDDINGS_DIR = Path("cache") / "jina-v5"
DATASET_DIR    = Path("dataset")
for d in (RESULTS_DIR, EMBEDDINGS_DIR, DATASET_DIR):
    d.mkdir(parents=True, exist_ok=True)

timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_PATH   = RESULTS_DIR / f"jina_benchmark_{timestamp}.csv"
S3_CHECKPOINT_KEY = f"{S3_PREFIX}/results/jina_benchmark_{timestamp}.csv"

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
# Model loading
# ---------------------------------------------------------------------------
print(f"\n🧠 Loading {MODEL_ID}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

# bfloat16 on GPU saves memory; float32 on CPU avoids BLAS fallback warnings
_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=_dtype,
)
model = model.to(device).eval()
print(f"   Loaded ✓  (dim={EMBED_DIM}, max_seq={MAX_SEQ_LEN})")


# ---------------------------------------------------------------------------
# Standard encoding: last-token pooling
# ---------------------------------------------------------------------------
def _last_token_pool(
    last_hidden_state: torch.Tensor,
    attention_mask:    torch.Tensor,
) -> torch.Tensor:
    """Last non-padding token pooling — the native pooling for Jina v5 / Qwen3."""
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx   = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_idx, seq_lengths]


def encode_texts(
    texts: list[str],
    is_query: bool = False,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Encode texts → (n, EMBED_DIM) float32, L2-normalized.
    Uses last-token pooling (Jina v5 native).
    """
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        pooled = _last_token_pool(out.last_hidden_state, inputs["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        all_vecs.append(pooled.cpu().numpy())

    return np.vstack(all_vecs)


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------
_Q_SCHEMA = pa.schema([
    ("question_id", pa.int32()),
    ("question",    pa.large_string()),
    ("embedding",   pa.list_(pa.float32(), EMBED_DIM)),
])

_C_SCHEMA = pa.schema([
    ("doc_id",     pa.int32()),
    ("chunk_idx",  pa.int32()),
    ("chunk_text", pa.large_string()),
    ("embedding",  pa.list_(pa.float32(), EMBED_DIM)),
])


def _emb_col(matrix: np.ndarray) -> pa.Array:
    """(n, EMBED_DIM) float32 numpy → pyarrow FixedSizeListArray, zero-copy."""
    flat = pa.array(matrix.flatten(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, EMBED_DIM)


def save_question_parquet(questions: list[str], path: Path, batch_size: int) -> Path:
    """Embed questions and stream to parquet."""
    writer = pq.ParquetWriter(str(path), _Q_SCHEMA, compression="snappy")
    total  = len(questions)
    for start in range(0, total, batch_size):
        batch = questions[start : start + batch_size]
        ids   = list(range(start, start + len(batch)))
        vecs  = encode_texts(batch, is_query=True, batch_size=batch_size)
        writer.write_table(pa.table({
            "question_id": pa.array(ids,   type=pa.int32()),
            "question":    pa.array(batch, type=pa.large_string()),
            "embedding":   _emb_col(vecs),
        }, schema=_Q_SCHEMA))
        print(f"    {start + len(batch):,}/{total:,}", end="\r")
    writer.close()
    print(f"    {total:,} questions ✓")
    return path


def save_recursive_parquet(
    corpus: list[str],
    chunker: "RecursiveChunker",
    path: Path,
    batch_size: int,
) -> tuple[Path, int]:
    """Chunk with RecursiveChunker, embed, stream to parquet."""
    writer       = pq.ParquetWriter(str(path), _C_SCHEMA, compression="snappy")
    total_chunks = 0
    buf_doc_ids  = []
    buf_ci       = []
    buf_texts    = []

    def _flush(force: bool = False):
        nonlocal buf_doc_ids, buf_ci, buf_texts
        if not buf_texts or (not force and len(buf_texts) < batch_size):
            return
        vecs = encode_texts(buf_texts, is_query=False, batch_size=batch_size)
        writer.write_table(pa.table({
            "doc_id":     pa.array(buf_doc_ids, type=pa.int32()),
            "chunk_idx":  pa.array(buf_ci,      type=pa.int32()),
            "chunk_text": pa.array(buf_texts,   type=pa.large_string()),
            "embedding":  _emb_col(vecs),
        }, schema=_C_SCHEMA))
        buf_doc_ids, buf_ci, buf_texts = [], [], []

    for doc_id, doc in enumerate(corpus):
        for ci, chunk_text in enumerate(chunker.chunk(doc)):
            buf_doc_ids.append(doc_id)
            buf_ci.append(ci)
            buf_texts.append(chunk_text)
            total_chunks += 1
            if len(buf_texts) >= batch_size:
                _flush()
                print(f"    doc {doc_id+1}/{len(corpus)}, {total_chunks:,} chunks", end="\r")

    _flush(force=True)
    writer.close()
    print(f"    {total_chunks:,} chunks ✓")
    return path, total_chunks


def save_late_parquet(
    corpus: list[str],
    chunker: "LateChunker",
    path: Path,
) -> tuple[Path, int]:
    """Chunk + embed with LateChunker (hidden-state mean pooling), stream to parquet."""
    writer       = pq.ParquetWriter(str(path), _C_SCHEMA, compression="snappy")
    total_chunks = 0

    for doc_id, doc in enumerate(corpus):
        chunk_texts, matrix = chunker.chunk_and_embed(doc)
        if not chunk_texts:
            continue
        doc_ids   = [doc_id] * len(chunk_texts)
        chunk_ids = list(range(len(chunk_texts)))
        writer.write_table(pa.table({
            "doc_id":     pa.array(doc_ids,     type=pa.int32()),
            "chunk_idx":  pa.array(chunk_ids,   type=pa.int32()),
            "chunk_text": pa.array(chunk_texts, type=pa.large_string()),
            "embedding":  _emb_col(matrix),
        }, schema=_C_SCHEMA))
        total_chunks += len(chunk_texts)
        print(f"    doc {doc_id+1}/{len(corpus)}, {total_chunks:,} chunks", end="\r")

    writer.close()
    print(f"    {total_chunks:,} chunks ✓")
    return path, total_chunks


def load_parquet(path: Path) -> tuple[list[str], np.ndarray]:
    """Load texts + embeddings from parquet. Returns (texts, float32 matrix)."""
    table    = pq.read_table(str(path))
    names    = table.schema.names
    text_col = next((c for c in ["chunk_text", "question", "text"] if c in names), None)
    if text_col is None:
        raise ValueError(f"No text column in {path.name}: {names}")
    texts    = table.column(text_col).to_pylist()
    flat_arr = table.column("embedding").combine_chunks().flatten()
    flat_np  = flat_arr.to_numpy(zero_copy_only=False).astype(np.float32)
    matrix   = flat_np.reshape(len(texts), EMBED_DIM)
    return texts, matrix


# ---------------------------------------------------------------------------
# Custom RecursiveChunker
# ---------------------------------------------------------------------------
class RecursiveChunker:
    """
    Recursive text splitter using a separator hierarchy.

    Tries each separator in order. Splits on the first one that divides
    the text, recurses on pieces still over chunk_size, then greedily
    merges small pieces back up to chunk_size. Falls back to token-level
    splitting when no separator applies.
    """

    def __init__(self, chunk_size: int, separators: list[str] = SEPARATORS):
        self.chunk_size = chunk_size
        self.separators = separators

    def _token_len(self, text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _merge(self, splits: list[str], sep: str) -> list[str]:
        """Greedily merge adjacent splits into chunks of at most chunk_size tokens."""
        merged   = []
        current  = []
        cur_len  = 0
        sep_len  = self._token_len(sep) if sep else 0

        for s in splits:
            s_len     = self._token_len(s)
            join_cost = sep_len if current else 0
            if cur_len + join_cost + s_len > self.chunk_size and current:
                merged.append(sep.join(current))
                current = [s]
                cur_len = s_len
            else:
                current.append(s)
                cur_len += join_cost + s_len

        if current:
            merged.append(sep.join(current))
        return merged

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if self._token_len(text) <= self.chunk_size:
            return [text]

        # Find the first separator that actually splits this text
        chosen_sep = separators[-1]
        remaining  = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                chosen_sep = sep
                remaining  = separators[i + 1:]
                break

        if chosen_sep == "":
            # No separator applies — hard split at the token level
            ids = tokenizer.encode(text, add_special_tokens=False)
            return [
                tokenizer.decode(ids[s : s + self.chunk_size], skip_special_tokens=True)
                for s in range(0, len(ids), self.chunk_size)
                if ids[s : s + self.chunk_size]
            ]

        pieces = [p for p in text.split(chosen_sep) if p.strip()]
        good   = []
        result = []

        for piece in pieces:
            if self._token_len(piece) <= self.chunk_size:
                good.append(piece)
            else:
                if good:
                    result.extend(self._merge(good, chosen_sep))
                    good = []
                result.extend(self._split(piece, remaining))

        if good:
            result.extend(self._merge(good, chosen_sep))

        return result

    def chunk(self, text: str) -> list[str]:
        return [c for c in self._split(text, self.separators) if c.strip()]


# ---------------------------------------------------------------------------
# Custom LateChunker
# ---------------------------------------------------------------------------
class LateChunker:
    """
    Late chunking: run the transformer on the full window, then mean-pool
    token hidden states within each chunk's span.

    Documents longer than MAX_SEQ_LEN are split into non-overlapping windows.
    Late chunking context is shared within each window.

    Chunks are produced by RecursiveChunker then grouped into windows.
    Character spans of each chunk within the concatenated window text are
    used to locate the corresponding token spans via offset_mapping.
    """

    def __init__(self, chunk_size: int):
        self.recursive  = RecursiveChunker(chunk_size)
        self.max_window = MAX_SEQ_LEN - 32  # safety margin for special tokens

    def chunk_and_embed(self, doc: str) -> tuple[list[str], np.ndarray]:
        """
        Returns (chunk_texts, embedding_matrix).
        embedding_matrix shape: (n_chunks, EMBED_DIM), dtype float32.
        """
        chunks = self.recursive.chunk(doc)
        if not chunks:
            return [], np.empty((0, EMBED_DIM), dtype=np.float32)

        all_embeddings = np.empty((len(chunks), EMBED_DIM), dtype=np.float32)

        # Precompute token lengths for windowing (no model calls — just tokenizer)
        tok_lens = [
            len(tokenizer.encode(c, add_special_tokens=False))
            for c in chunks
        ]

        i = 0
        while i < len(chunks):
            # Build the largest window starting at chunk i
            window_chunks = []
            window_tok_len = 0
            j = i
            while j < len(chunks):
                if window_tok_len + tok_lens[j] + 2 > self.max_window:
                    break
                window_chunks.append(chunks[j])
                window_tok_len += tok_lens[j]
                j += 1

            if not window_chunks:
                # Single chunk exceeds max_window — fall back to standard encode
                vec = encode_texts([chunks[i]], is_query=False, batch_size=1)
                all_embeddings[i] = vec[0]
                i += 1
                continue

            # Concatenate chunks into a single window text
            window_text = _WIN_SEP.join(window_chunks)

            # Tokenize with character offset mapping
            inputs     = tokenizer(
                window_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                max_length=self.max_window,
                truncation=True,
                add_special_tokens=True,
            )
            offset_map = inputs.pop("offset_mapping")[0].tolist()
            inputs     = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out    = model(**inputs)
            hidden = out.last_hidden_state[0]  # (seq_len, EMBED_DIM)

            # For each chunk, locate its character span in window_text,
            # map to token indices, and mean-pool hidden states
            char_pos = 0
            for k_idx, chunk_text in enumerate(window_chunks):
                chunk_char_start = char_pos
                chunk_char_end   = char_pos + len(chunk_text)

                tok_s = None
                tok_e = None
                for ti, (cs, ce) in enumerate(offset_map):
                    if cs == ce:
                        continue  # special token — no character coverage
                    if tok_s is None and ce > chunk_char_start:
                        tok_s = ti
                    if cs < chunk_char_end:
                        tok_e = ti + 1

                if tok_s is None or tok_e is None or tok_s >= tok_e:
                    tok_s, tok_e = 0, hidden.shape[0]

                span   = hidden[tok_s:tok_e]
                if span.shape[0] == 0:
                    span = hidden

                pooled = span.mean(dim=0)
                pooled = torch.nn.functional.normalize(
                    pooled.float().unsqueeze(0), p=2, dim=1
                ).squeeze()
                all_embeddings[i + k_idx] = pooled.cpu().numpy()

                # Advance past this chunk + the join separator
                char_pos += len(chunk_text) + len(_WIN_SEP)

            i = j

        return chunks, all_embeddings


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
print("\n📥 Loading dataset...")
corpus_local    = DATASET_DIR / "corpus.parquet"
questions_local = DATASET_DIR / "questions.parquet"

if not corpus_local.exists() or not questions_local.exists():
    corpus_ok    = s3_download("sec-benchmark/dataset/corpus.parquet",    corpus_local)
    questions_ok = s3_download("sec-benchmark/dataset/questions.parquet", questions_local)
    if corpus_ok and questions_ok:
        print("  Loaded from S3.")
    else:
        print("  S3 miss — loading from HuggingFace...")
        from datasets import load_dataset as _load_hf
        _ds_c = _load_hf("Tim-Pinecone/sec-10k-qa", "corpus",    split="train")
        _ds_q = _load_hf("Tim-Pinecone/sec-10k-qa", "questions", split="train")
        _ds_c.to_parquet(str(corpus_local))
        _ds_q.to_parquet(str(questions_local))
        s3_upload(corpus_local,    "sec-benchmark/dataset/corpus.parquet")
        s3_upload(questions_local, "sec-benchmark/dataset/questions.parquet")
        print("  Loaded from HuggingFace and cached to S3.")

from datasets import load_dataset
ds_corpus    = load_dataset("parquet", data_files=str(corpus_local),    split="train")
ds_questions = load_dataset("parquet", data_files=str(questions_local), split="train")

corpus            = [row["text"]               for row in ds_corpus]
questions         = [row["question"]           for row in ds_questions]
relevant_passages = [row["chunk_must_contain"] for row in ds_questions]

if args.dry_run:
    # Truncate the first document to ~3000 chars so each chunker config
    # produces only a handful of chunks — fast on CPU, still exercises every
    # code path including LateChunker's hidden-state span pooling.
    corpus            = [corpus[0][:3000]]
    questions         = questions[:5]
    relevant_passages = relevant_passages[:5]
    print(f"  🔬 Dry run: 1 doc (truncated to 3000 chars), {len(questions)} questions")
else:
    print(f"  {len(corpus)} documents, {len(questions)} QA pairs")

if not questions:
    raise SystemExit("❌ No questions loaded.")


# ---------------------------------------------------------------------------
# Pure-numpy evaluation
# ---------------------------------------------------------------------------
def evaluate(
    q_matrix:         np.ndarray,
    c_matrix:         np.ndarray,
    chunk_texts:      list[str],
    relevant_passages: list[str],
    k_values:         list[int],
) -> dict:
    """Cosine similarity retrieval with substring-match ground truth."""
    Q     = q_matrix / (np.linalg.norm(q_matrix, axis=1, keepdims=True) + 1e-9)
    C     = c_matrix / (np.linalg.norm(c_matrix, axis=1, keepdims=True) + 1e-9)
    sim   = Q @ C.T                                      # (n_q, n_c)
    max_k = max(k_values)
    top_i = np.argsort(-sim, axis=1)[:, :max_k]          # (n_q, max_k)

    ranks = []
    for q_idx, relevant in enumerate(relevant_passages):
        rank = 0
        for r, ci in enumerate(top_i[q_idx], 1):
            if relevant in chunk_texts[ci]:
                rank = r
                break
        ranks.append(rank)

    out = {}
    for k in k_values:
        out[f"R@{k}"]   = sum(1 for r in ranks if 0 < r <= k) / len(ranks)
        out[f"MRR@{k}"] = float(np.mean([1 / r if 0 < r <= k else 0.0 for r in ranks]))
    return out


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------
completed = set()
results   = []

try:
    objs = _s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/results/").get("Contents", [])
    csvs = sorted([o["Key"] for o in objs if o["Key"].endswith(".csv")], reverse=True)
    if csvs:
        local_resume = RESULTS_DIR / "jina_resume.csv"
        if s3_download(csvs[0], local_resume):
            df_prev = pd.read_csv(local_resume)
            results = df_prev.to_dict("records")
            for row in df_prev.itertuples():
                if not (hasattr(row, "error") and pd.notna(getattr(row, "error", None))):
                    completed.add((str(row.chunker), int(row.chunk_size)))
            print(f"\n📂 Resuming: {len(completed)} configs already done (from {csvs[0]})")
            CHECKPOINT_PATH   = local_resume
            S3_CHECKPOINT_KEY = csvs[0]
except Exception as e:
    print(f"  (no checkpoint found: {e})")


# ---------------------------------------------------------------------------
# Phase 1 — Embed questions
# ---------------------------------------------------------------------------
q_parquet_path = EMBEDDINGS_DIR / f"questions_{MODEL_KEY}.parquet"
q_s3_key       = f"{S3_PREFIX}/embeddings/questions_{MODEL_KEY}.parquet"

if s3_exists(q_s3_key):
    print(f"\n✅ Question embeddings already on S3: {q_parquet_path.name}")
    if not q_parquet_path.exists():
        s3_download(q_s3_key, q_parquet_path)
else:
    print(f"\n🔢 Phase 1 — Embedding {len(questions)} questions...")
    save_question_parquet(questions, q_parquet_path, batch_size=args.batch_size)
    s3_upload(q_parquet_path, q_s3_key)

_, q_matrix = load_parquet(q_parquet_path)
print(f"   Question matrix: {q_matrix.shape}")


# ---------------------------------------------------------------------------
# Phase 2 — Per-config: chunk, embed, evaluate
# ---------------------------------------------------------------------------
total_configs = len(CONFIGS)
print(f"\n{'='*60}")
print(f"Phase 2 — {total_configs} configs  |  model: {MODEL_KEY}")
print(f"{'='*60}")

for cfg_idx, (chunker_name, chunk_size) in enumerate(CONFIGS, 1):
    run_key = (chunker_name, chunk_size)
    if run_key in completed:
        print(f"\n[{cfg_idx}/{total_configs}] ⏭️  {chunker_name}(size={chunk_size}) — already done")
        continue

    print(f"\n[{cfg_idx}/{total_configs}] {chunker_name}(size={chunk_size})")
    t0  = time.time()
    row = {
        "chunker":         chunker_name,
        "chunk_size":      chunk_size,
        "embedding_model": MODEL_KEY,
    }

    try:
        chunk_parquet_path = EMBEDDINGS_DIR / f"chunks_{chunker_name}_{chunk_size}_{MODEL_KEY}.parquet"
        chunk_s3_key       = f"{S3_PREFIX}/embeddings/chunks_{chunker_name}_{chunk_size}_{MODEL_KEY}.parquet"

        if s3_exists(chunk_s3_key):
            print(f"  ✅ Chunk embeddings already on S3")
            if not chunk_parquet_path.exists():
                s3_download(chunk_s3_key, chunk_parquet_path)
            chunk_texts, c_matrix = load_parquet(chunk_parquet_path)
            print(f"  Loaded {len(chunk_texts):,} chunks")
        else:
            if chunker_name == "RecursiveChunker":
                print(f"  Chunking + embedding corpus...")
                chunker = RecursiveChunker(chunk_size)
                save_recursive_parquet(corpus, chunker, chunk_parquet_path, batch_size=args.batch_size)
            elif chunker_name == "LateChunker":
                print(f"  Chunking + embedding corpus (late chunking, 32K windows)...")
                chunker = LateChunker(chunk_size)
                save_late_parquet(corpus, chunker, chunk_parquet_path)
            else:
                raise ValueError(f"Unknown chunker: {chunker_name}")

            s3_upload(chunk_parquet_path, chunk_s3_key)
            chunk_texts, c_matrix = load_parquet(chunk_parquet_path)

        row["n_chunks"] = len(chunk_texts)
        print(f"  ▶️  Evaluating {len(chunk_texts):,} chunks vs {len(questions)} questions...")

        metrics = evaluate(q_matrix, c_matrix, chunk_texts, relevant_passages, K_VALUES)
        elapsed = time.time() - t0

        row["time_sec"] = round(elapsed, 1)
        row.update(metrics)

        recall_str = "  ".join(f"R@{k}={metrics[f'R@{k}']:.1%}" for k in K_VALUES)
        print(f"  ✅  {recall_str}  ({elapsed:.0f}s)")

    except Exception as e:
        elapsed        = time.time() - t0
        row["time_sec"] = round(elapsed, 1)
        row["error"]    = str(e)
        print(f"  ❌  {e} ({elapsed:.0f}s)")
        import traceback
        traceback.print_exc()

    results.append(row)
    completed.add(run_key)
    pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
    s3_upload(CHECKPOINT_PATH, S3_CHECKPOINT_KEY)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)
if "error" in df.columns:
    df_ok = df[df["error"].isna()].copy()
else:
    df_ok = df.copy()

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
show_cols = ["chunker", "chunk_size", "n_chunks"] + [f"R@{k}" for k in K_VALUES] + ["time_sec"]
show_cols = [c for c in show_cols if c in df.columns]
print(df[show_cols].to_string(index=False))

print(f"\n💾 {CHECKPOINT_PATH}")
print(f"☁️  s3://{S3_BUCKET}/{S3_CHECKPOINT_KEY}")
print(f"\n📁 Embedding parquets:")
for p in sorted(EMBEDDINGS_DIR.glob("*.parquet")):
    print(f"  {p.name}  ({p.stat().st_size / 1_048_576:.1f} MB)")
