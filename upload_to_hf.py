"""
Upload the SEC 10-K QA dataset to HuggingFace Hub.

Usage:
    uv run python upload_to_hf.py
    uv run python upload_to_hf.py --dataset-dir ./dataset --repo Tim-Pinecone/sec-10k-qa
"""

import argparse
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("❌ HF_TOKEN not set. Add it to .env")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="./dataset", help="Path to dataset/ folder")
parser.add_argument("--repo", default=None, help="HF repo id, e.g. Tim-Pinecone/sec-10k-qa")
args = parser.parse_args()

DATASET_DIR = Path(args.dataset_dir)

# Resolve username and repo name
api = HfApi(token=HF_TOKEN)
username = api.whoami()["name"]
repo_id = args.repo or f"{username}/sec-10k-qa"

print(f"  HuggingFace user : {username}")
print(f"  Target repo      : {repo_id}")
print(f"  Dataset dir      : {DATASET_DIR}")

# ---------------------------------------------------------------------------
# Load corpus.jsonl
# ---------------------------------------------------------------------------
corpus_path = DATASET_DIR / "corpus.jsonl"
if not corpus_path.exists():
    raise SystemExit(f"❌ {corpus_path} not found. Download from S3 first.")

corpus_rows = []
with open(corpus_path) as f:
    for line in f:
        doc = json.loads(line)
        corpus_rows.append({"document_id": doc["id"], "text": doc["text"]})

print(f"\n  Loaded {len(corpus_rows)} documents from corpus.jsonl")

# ---------------------------------------------------------------------------
# Load sec_dataset.jsonl
# ---------------------------------------------------------------------------
qa_path = DATASET_DIR / "sec_dataset.jsonl"
if not qa_path.exists():
    raise SystemExit(f"❌ {qa_path} not found. Download from S3 first.")

qa_rows = []
with open(qa_path) as f:
    for line in f:
        record = json.loads(line)
        doc_id = record.get("document_id")
        for sample in record.get("samples", []):
            qa_rows.append({
                "question":          sample.get("question", ""),
                "answer":            sample.get("answer", ""),
                "chunk_must_contain": sample.get("chunk_must_contain", ""),
                "document_id":       doc_id,
            })

print(f"  Loaded {len(qa_rows)} QA pairs from sec_dataset.jsonl")

if not qa_rows:
    raise SystemExit("❌ No QA pairs found. Is the generate step complete?")

print(f"\n  corpus:    {len(corpus_rows)} rows")
print(f"  questions: {len(qa_rows)} rows")

# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------
CARD = f"""\
---
license: mit
task_categories:
- question-answering
- text-retrieval
language:
- en
tags:
- sec
- 10-k
- rag
- chunking
- mtcb
- finance
pretty_name: SEC 10-K QA (MTCB)
size_categories:
- 1K<n<10K
configs:
- config_name: corpus
  data_files:
  - split: train
    path: data/corpus/train-00000-of-00001.parquet
- config_name: questions
  data_files:
  - split: train
    path: data/questions/train-00000-of-00001.parquet
---

# SEC 10-K QA Dataset

A retrieval QA dataset built from SEC 10-K annual filings, designed for benchmarking
RAG chunking strategies with [MTCB](https://github.com/chonkie-inc/mtcb).

## Contents

| Split | Rows | Description |
|-------|------|-------------|
| `corpus` | {len(corpus_rows)} | Cleaned 10-K filing text (20 companies × 5 years) |
| `questions` | {len(qa_rows)} | QA pairs generated from corpus chunks |

## Companies

AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, JNJ, UNH, V, PG,
NVDA, META, BRK, XOM, WMT, BAC, PFE, DIS, NFLX, AMD

## Schema

**corpus**
- `document_id` — filing identifier (ticker + accession number)
- `text` — cleaned filing text

**questions**
- `question` — question about a passage in the filing
- `answer` — answer to the question
- `chunk_must_contain` — verbatim excerpt from the source chunk (ground truth for retrieval)
- `document_id` — links back to corpus

## Usage with MTCB

```python
from datasets import load_dataset
from mtcb import SimpleEvaluator

ds = load_dataset("{repo_id}")
corpus    = [row["text"] for row in ds["corpus"]]
questions = [row["question"] for row in ds["questions"]]
passages  = [row["chunk_must_contain"] for row in ds["questions"]]
```
"""

# ---------------------------------------------------------------------------
# Push to hub — upload raw Parquet files to avoid cross-split schema validation
# ---------------------------------------------------------------------------
print(f"\n  Pushing to https://huggingface.co/datasets/{repo_id} ...")
api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

def _upload_parquet(rows, path_in_repo):
    table = pa.Table.from_pylist(rows)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
    )

_upload_parquet(corpus_rows, "data/corpus/train-00000-of-00001.parquet")
print(f"  ✅ corpus uploaded")
_upload_parquet(qa_rows, "data/questions/train-00000-of-00001.parquet")
print(f"  ✅ questions uploaded")

api.upload_file(
    path_or_fileobj=CARD.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
    token=HF_TOKEN,
)

print(f"\n  ✅ Done: https://huggingface.co/datasets/{repo_id}")
