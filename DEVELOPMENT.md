# Better Chunking Benchmarks

Notes on bottlenecks, bugs, and ideas to make the SEC chunking benchmark faster and easier to iterate on.

---

## Issues Encountered

### 1. `embed_batch(input_type=...)` incompatibility
MTCB's `SimpleEvaluator` calls `embed_batch(input_type=...)` internally, but chonkie's `SentenceTransformerEmbeddings` doesn't accept that kwarg. Crashes immediately at evaluation time.

**Fix (applied):** Wrap `SentenceTransformerEmbeddings` to drop all extra kwargs — MTCB passes `input_type`, `show_progress`, and possibly others that chonkie doesn't accept:
```python
class _Embeddings(SentenceTransformerEmbeddings):
    def embed_batch(self, texts, **kwargs):
        return super().embed_batch(texts)
```
This should be upstreamed to MTCB or chonkie.

---

### 2. ~10 minute setup time on every run
Every deploy installs uv, Python, PyTorch (CUDA), sentence-transformers, mtcb, chonkie, datasets from scratch. This is the biggest time sink — the benchmark itself is fast but you wait 10+ minutes before it even starts.

**Options to fix:**
- **Bake a Docker image** with all deps pre-installed. Push to ECR. EC2 pulls and runs it — same pattern as sec-dataset-builder. Setup drops to ~30 seconds.
- **Use an S3-cached venv** — after first run, tar the `.venv` to S3. Subsequent runs download and unpack it (~60s) instead of reinstalling (~10 min).
- **Persistent instance** — don't terminate after each run. SSH/SSM in to rerun. Costs money while idle but good for iteration.

**Recommended:** Docker image baked with all deps. Re-run a benchmark = 30s pull + actual benchmark time.

---

### 3. No embedding cache between runs
Every run re-embeds all 95 corpus documents and 950 questions from scratch. For `bge-large-en-v1.5` this takes a few minutes even on GPU. When iterating on chunkers, the question embeddings never change — only the corpus chunking changes.

**Fix:** MTCB has a `cache` object on `SimpleEvaluator`. Pre-compute question embeddings once, serialize to disk (numpy `.npy` or parquet), reload on subsequent runs. Only re-chunk + re-embed the corpus per run.

---

### 4. One chunker per EC2 run
Currently `benchmark.py` runs exactly one chunker configuration. To compare strategies you need to deploy, wait, get results, redeploy with different params. Very slow iteration loop.

**Fix:** Run all configurations in a single job. Pass a list of `(chunker_name, chunk_size)` pairs and loop. Results land in one CSV. One EC2 bill instead of N.

Example matrix:
```python
CONFIGS = [
    ("RecursiveChunker", 256),
    ("RecursiveChunker", 512),
    ("RecursiveChunker", 1024),
    ("SentenceChunker", 512),
    ("TokenChunker", 512),
]
```

---

### 5. Dataset loaded from HuggingFace on every run
`load_dataset("Tim-Pinecone/sec-10k-qa")` hits HF's servers each time. Slow and fragile (HF outage = broken benchmark).

**Fix:** Cache the dataset in S3 as parquet. On EC2, download from S3 instead. Already have the parquet files from `upload_to_hf.py` — just upload them to `s3://mtcb-benchmark/sec-benchmark/dataset/` and `load_dataset` from local path.

---

### 6. SSO token expires mid-session
`aws sso login` sessions expire after a few hours. Running `./deploy.sh` fails silently on credential errors unless you check the output carefully.

**Fix:** Add a credential check at the top of `deploy.sh` before running Terraform:
```bash
aws sts get-caller-identity --query Account --output text || {
  echo "AWS credentials expired. Run: aws sso login --profile $AWS_PROFILE"
  exit 1
}
```

---

### 7. No checkpoint / resume
If the benchmark crashes mid-run (OOM, spot interruption, etc.) the whole run is lost. The `sec-dataset-builder.py` had a checkpoint CSV pattern — `benchmark.py` doesn't.

**Fix:** Write results to a checkpoint CSV after each chunker run. On restart, skip already-completed `(chunker, chunk_size)` pairs.

---

## Proposed Architecture for Fast Iteration

```
benchmark.py (improved)
├── Load dataset from S3 (not HF)
├── Load/save question embeddings cache from S3
├── Loop over all chunker × chunk_size configs
│   ├── Skip if already in checkpoint CSV
│   ├── Chunk corpus
│   ├── Embed chunks
│   ├── Evaluate R@k + MRR@k
│   └── Append to checkpoint CSV + sync to S3
└── Upload final results CSV to S3
```

Docker image (pre-baked):
```
FROM python:3.11-slim  (or CUDA base)
RUN install torch + sentence-transformers + mtcb + chonkie + datasets
# No compilation at runtime — everything pre-built
```

Total time per run with these fixes: **~5 min** (image pull + embed questions once + loop chunkers) vs **~25+ min** currently.

---

## Baseline Results

**Run date:** 2026-04-06  
**Dataset:** Tim-Pinecone/sec-10k-qa (95 docs, 950 QA pairs, SEC 10-K filings)  
**Instance:** g4dn.xlarge (T4 GPU, us-east-1)  
**Setup time:** ~10 min | **Eval time:** 1828s (~30 min)

| Chunker | Chunk Size | Embedding Model | R@1 | R@3 | R@5 | R@10 | MRR@10 | Time |
|---------|-----------|-----------------|-----|-----|-----|------|--------|------|
| RecursiveChunker | 1024 | bge-large-en-v1.5 | 25.2% | 37.0% | 42.0% | 48.5% | 0.324 | 1828s |

**Reading the numbers:** R@10 = 48.5% means the answer chunk was in the top 10 retrieved results for ~half the questions. For a chunking benchmark the number to watch is R@5 — a good chunker on a good dataset should be pushing 60%+. 48.5% at R@10 with chunk_size=1024 gives us a clear baseline to beat with smaller chunks or smarter chunkers.

**Next runs to try:**
- `chunk_size=256` and `chunk_size=512` — smaller chunks may improve recall on this dataset since questions target specific passages
- `SentenceChunker` and `TokenChunker` at the same sizes for comparison
- `SemanticChunker` if GPU memory allows

---

## Research Quality Issues

Notes from a critical review of the methodology. These should be addressed before publishing or sharing findings externally.

---

### 1. Chunker comparison is not apples-to-apples (critical)

The `chunk_size` parameter means different things to different chunkers. Observed chunk counts at the same parameter value:

| Chunker | Size param | Actual chunks |
|---|---|---|
| RecursiveChunker | 512 | 91,897 |
| LateChunker | 512 | 17,297 |
| SemanticChunker | 512 | 85,156 |
| SemanticChunker | 1024 | 84,624 |

LateChunker at `chunk_size=512` produces the same number of chunks as RecursiveChunker at well above 1024. SemanticChunker ignores the size parameter almost entirely. Comparing by parameter value rather than actual average chunk length is comparing different things.

**Fix:** Report average chunk length (tokens) alongside chunk count. Compare chunkers at matched average lengths, not matched parameter values.

---

### 2. Margin between top chunkers is not statistically significant

RecursiveChunker/1024 vs LateChunker/512 at R@10: 50.0% vs 50.1%. With 950 questions, that is a difference of ~1 question. No significance testing has been run.

**Fix:** Run McNemar's test on paired per-question results before claiming one chunker beats another. Results within ~1% should be reported as a tie until significance is established.

---

### 3. Ground truth uses substring match

`relevant_passage in chunk.text` passes if any substring of the passage appears in the chunk. A very short `chunk_must_contain` (e.g. a dollar figure) would match almost any financially-relevant chunk. A long one may never match even with correct retrieval. The length distribution of ground truth passages has not been analysed.

**Fix:** Audit `chunk_must_contain` length distribution. Consider a minimum length threshold. Optionally replace substring match with a token overlap (F1) metric.

---

### 4. LLM-generated questions have selection bias

Gemini generated both the questions and the `chunk_must_contain` passages. It likely selected salient, extractable passages rather than buried or implicit information. This makes the benchmark easier than real-world RAG and may inflate absolute scores.

**Fix:** Supplement with human-written questions, or use a second LLM with a different prompt to generate adversarial questions targeting less prominent passages.

---

### 5. 10-K boilerplate inflates scores

Annual 10-K filings repeat large sections year-over-year (risk factors, business descriptions). A question about Apple's revenue recognition policy may have near-identical ground truth passages in 5 different filings. Retrieval appears better than it is because the answer exists in multiple places in the corpus.

**Fix:** Deduplicate near-identical passages across filings before generating QA pairs. Or measure per-company rather than pooled.

---

### 6. Only one embedding model tested

All chunker comparisons were run against `text-embedding-ada-002`. The relative ranking of chunkers may change with a different embedding model. The interaction between chunking strategy and embedding model is uncontrolled.

**Fix:** Re-run the same chunker matrix with at least one other embedding model (e.g. `text-embedding-3-large` or a local model) to test whether rankings are stable.

---

### 7. Chunk overlap not tested

Chunking with overlap (e.g. 20% stride) is standard in production RAG and consistently improves retrieval. The current benchmark only tests non-overlapping chunks, which may understate what RecursiveChunker and SentenceChunker can achieve.

**Fix:** Add overlap variants to the config matrix (e.g. RecursiveChunker/1024 with 20% overlap).

---

### Bottom line

The finding that **larger chunk sizes outperform smaller ones** is robust. The finding that **RecursiveChunker is the best specific strategy** is plausible but not conclusive — the margin over LateChunker is within noise and the comparison does not control for actual chunk size. The strongest defensible claim is: use chunk_size=1024, use RecursiveChunker as a safe default, and don't pay the cost of semantic or neural chunkers for marginal gains.

---

## Quick Wins (in order of impact)

| # | Change | Time saved |
|---|--------|-----------|
| 1 | Docker image with pre-baked deps | ~10 min/run |
| 2 | Run all chunkers in one job | N-1 deploys avoided |
| 3 | Cache question embeddings to S3 | ~2 min/run |
| 4 | Load dataset from S3 not HF | ~30s/run |
| 5 | Checkpoint/resume | saves full re-runs on crash |
| 6 | Credential check in deploy.sh | avoids confusing failures |
