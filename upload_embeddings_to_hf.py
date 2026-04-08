"""
Upload SEC 10-K embedding parquets from S3 to HuggingFace.

Downloads all embedding parquets from S3 to a local staging directory,
writes a dataset card with config declarations, then uploads via the HF Hub.
Resumable — safe to re-run if interrupted.

Usage:
    uv run python upload_embeddings_to_hf.py
    uv run python upload_embeddings_to_hf.py --repo Tim-Pinecone/sec-10k-qa-embeddings
    uv run python upload_embeddings_to_hf.py --s3-prefix sec-benchmark/embeddings --dry-run
"""

import argparse
import os
import subprocess
from pathlib import Path

import boto3
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--repo",        default="Tim-Pinecone/sec-10k-qa-embeddings")
parser.add_argument("--s3-bucket",   default=os.environ.get("S3_BUCKET", "mtcb-benchmark"))
parser.add_argument("--s3-prefix",   default="sec-benchmark/embeddings")
parser.add_argument("--staging-dir", default="staging/embeddings")
parser.add_argument("--dry-run",     action="store_true")
args = parser.parse_args()

STAGING = Path(args.staging_dir)
STAGING.mkdir(parents=True, exist_ok=True)
s3 = boto3.client("s3")

# ---------------------------------------------------------------------------
# Step 1 — Sync parquets from S3 to local staging
# ---------------------------------------------------------------------------
print(f"📥 Syncing s3://{args.s3_bucket}/{args.s3_prefix}/ → {STAGING}/")
result = subprocess.run([
    "aws", "s3", "sync",
    f"s3://{args.s3_bucket}/{args.s3_prefix}/",
    str(STAGING),
    "--exclude", "*",
    "--include", "*.parquet",
], check=True)

parquets = sorted(STAGING.glob("*.parquet"))
print(f"   {len(parquets)} parquet files staged:")
for p in parquets:
    print(f"   {p.name}  ({p.stat().st_size / 1_048_576:.1f} MB)")

# ---------------------------------------------------------------------------
# Step 2 — Build HF dataset card with config declarations
# ---------------------------------------------------------------------------
def parquet_to_config_name(fname: str) -> str:
    """
    questions_text-embedding-ada-002.parquet  → questions_ada002
    chunks_LateChunker_512_text-embedding-ada-002.parquet → chunks_LateChunker_512_ada002
    """
    stem  = fname.replace(".parquet", "")
    # Normalise model suffix
    stem  = stem.replace("text-embedding-ada-002", "ada002")
    stem  = stem.replace("text-embedding-3-large", "oai3large")
    stem  = stem.replace("bge-large-en-v1.5",      "bge_large")
    # Make HF-safe (no hyphens in config names)
    return stem.replace("-", "_")


configs_yaml = ""
for p in parquets:
    config_name = parquet_to_config_name(p.name)
    configs_yaml += f"""- config_name: {config_name}
  data_files:
  - split: train
    path: data/{p.name}
"""

readme = f"""---
license: apache-2.0
task_categories:
- feature-extraction
language:
- en
tags:
- embeddings
- chunking
- RAG
- SEC
- 10-K
configs:
{configs_yaml}
---

# SEC 10-K QA Embeddings

Pre-computed embeddings for the [Tim-Pinecone/sec-10k-qa](https://huggingface.co/datasets/Tim-Pinecone/sec-10k-qa) dataset.

## What's in here

Each config is a parquet file containing pre-computed `text-embedding-ada-002` embeddings
for a specific chunking strategy applied to the SEC 10-K corpus.

| Config | Description |
|--------|-------------|
| `questions_ada002` | All 950 evaluation questions |
| `chunks_RecursiveChunker_512_ada002` | RecursiveChunker at chunk_size=512 |
| `chunks_RecursiveChunker_1024_ada002` | RecursiveChunker at chunk_size=1024 |
| `chunks_SentenceChunker_512_ada002` | SentenceChunker at chunk_size=512 |
| `chunks_SentenceChunker_1024_ada002` | SentenceChunker at chunk_size=1024 |
| `chunks_SemanticChunker_512_ada002` | SemanticChunker at chunk_size=512 |
| `chunks_SemanticChunker_1024_ada002` | SemanticChunker at chunk_size=1024 |
| `chunks_LateChunker_512_ada002` | LateChunker at chunk_size=512 |
| `chunks_LateChunker_1024_ada002` | LateChunker at chunk_size=1024 |
| `chunks_NeuralChunker_0_ada002` | NeuralChunker (auto chunk size) |

## Schema

**Questions parquet:**
```
question_id  int32
question     string
embedding    list<float32>[1536]
```

**Chunk parquets:**
```
doc_id       int32
chunk_idx    int32
chunk_text   string
embedding    list<float32>[1536]
```

## Corpus

20 large-cap US companies (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, etc.),
5 annual 10-K filings each = 95 documents, 950 QA pairs.

## Usage

```python
from datasets import load_dataset

# Load question embeddings
questions = load_dataset("Tim-Pinecone/sec-10k-qa-embeddings", "questions_ada002", split="train")

# Load chunk embeddings for a specific chunker
chunks = load_dataset("Tim-Pinecone/sec-10k-qa-embeddings", "chunks_LateChunker_512_ada002", split="train")
```
"""

readme_path = STAGING / "README.md"
readme_path.write_text(readme)
print(f"\n📝 Dataset card written → {readme_path}")

# ---------------------------------------------------------------------------
# Step 3 — Create repo if needed and upload
# ---------------------------------------------------------------------------
if args.dry_run:
    print(f"\n🔍 Dry run — would upload {len(parquets)} parquets to {args.repo}")
    print("   Remove --dry-run to upload.")
else:
    api   = HfApi(token=os.environ.get("HF_TOKEN"))
    token = os.environ.get("HF_TOKEN")

    # Create dataset repo if it doesn't exist
    try:
        api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
        print(f"\n✅ Repo ready: https://huggingface.co/datasets/{args.repo}")
    except Exception as e:
        print(f"⚠️  Could not create repo: {e}")

    # Upload — hf upload-large-folder handles chunking + resumable uploads
    print(f"\n🤗 Uploading to {args.repo}...")
    subprocess.run([
        "hf", "upload", args.repo,
        str(STAGING),
        "--repo-type", "dataset",
        "--token", token,
    ], check=True)

    print(f"\n✅ Done: https://huggingface.co/datasets/{args.repo}")
