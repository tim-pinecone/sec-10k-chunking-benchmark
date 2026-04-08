# SEC 10-K Chunking Benchmark

A retrieval benchmark comparing chunking strategies on real-world SEC 10-K filings. Tests how well different chunking approaches support RAG-style retrieval using OpenAI embeddings.

**Dataset:** [Tim-Pinecone/sec-10k-qa](https://huggingface.co/datasets/Tim-Pinecone/sec-10k-qa) — 95 documents, 950 QA pairs  
**Embeddings:** [Tim-Pinecone/sec-10k-qa-embeddings](https://huggingface.co/datasets/Tim-Pinecone/sec-10k-qa-embeddings) — pre-computed per chunker

---

## Results

Full results with R@10–R@30 are in [RESULTS.md](RESULTS.md).

| Chunker | Chunk Size | R@10 | R@30 | Chunks | Time |
|---------|-----------|------|------|--------|------|
| **RecursiveChunker** | **1024** | **50.0%** | **55.7%** | 42,830 | 247s |
| LateChunker | 512 | 50.1% | 56.2% | 17,297 | 5,058s |
| SentenceChunker | 1024 | 46.4% | 52.5% | 35,900 | 244s |
| NeuralChunker | auto | 44.9% | 51.4% | 73,497 | 2,108s |
| SemanticChunker | 1024 | 40.6% | 46.4% | 84,624 | 209s |

**Bottom line:** `RecursiveChunker` at `chunk_size=1024` matches or beats every other strategy on this dataset, runs in under 5 minutes, and requires no GPU for chunking. The margin over LateChunker is within noise — see [DEVELOPMENT.md](DEVELOPMENT.md) for caveats.

---

## Corpus

20 large-cap US companies × 5 annual 10-K filings = 95 documents:

AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V, JNJ, WMT, XOM, PG, MA, HD, BAC, ABBV, MRK, CVX

QA pairs were generated with Gemini 1.5 Pro — 10 questions per document, each with a `chunk_must_contain` ground truth passage for substring-match evaluation.

---

## Repository Structure

```
sec-benchmark/
├── benchmark.py              # Main benchmark — chunks, embeds, evaluates
├── reeval.py                 # Re-evaluate at higher k from saved parquets
├── sec-dataset-builder.py    # Builds the corpus + QA dataset from EDGAR
├── upload_to_hf.py           # Uploads dataset to HuggingFace
├── upload_embeddings_to_hf.py# Uploads pre-computed embeddings to HuggingFace
├── requirements.txt          # Python dependencies
├── Dockerfile                # Benchmark image (CUDA, for EC2)
├── Dockerfile.dataset        # Dataset builder image (CPU)
├── RESULTS.md                # Final results table
├── DEVELOPMENT.md            # Architecture notes, known issues, research caveats
├── results/                  # Raw CSV output from benchmark runs
└── infra/                    # Terraform + deploy script for AWS EC2
    ├── deploy.sh             # One-command deploy
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    └── user_data.sh.tpl
```

---

## Running the Benchmark

### Prerequisites

- Docker (for local builds)
- AWS CLI with SSO configured (`aws sso login --profile <profile>`)
- An S3 bucket (set `S3_BUCKET` in `.env`)
- OpenAI API key (for `text-embedding-ada-002`)

### Setup

```bash
cp .env.example .env
# Fill in AWS_PROFILE, S3_BUCKET, OPENAI_API_KEY, HF_TOKEN
```

### Deploy to EC2

```bash
cd infra
./deploy.sh apply
```

This will:
1. Check AWS credentials
2. Build and push the Docker image to ECR
3. Upload the dataset parquets to S3 (if not already there)
4. Provision a `g4dn.xlarge` (T4 GPU) via Terraform
5. Run the benchmark and upload results to S3

Results land at `s3://<S3_BUCKET>/sec-benchmark/<run-name>/results/`.

### Configure what to run

Set `BENCHMARK_ARGS` in `.env`:

```bash
# Single chunker
BENCHMARK_ARGS=--embedding-model openai:text-embedding-ada-002 --configs RecursiveChunker:1024

# Full matrix
BENCHMARK_ARGS=--embedding-model openai:text-embedding-ada-002 \
  --configs RecursiveChunker:512,RecursiveChunker:1024,SentenceChunker:1024,LateChunker:512 \
  --run-name my-experiment
```

Available chunkers: `RecursiveChunker`, `SentenceChunker`, `TokenChunker`, `SemanticChunker`, `LateChunker`, `NeuralChunker`

### Re-evaluate at higher k (no re-embedding)

```bash
python reeval.py \
  --model text-embedding-ada-002 \
  --k 10,20,30 \
  --out results/reeval.csv
```

Downloads pre-computed embeddings from S3 and evaluates without making any API calls.

---

## Building the Dataset

The dataset is already published at [Tim-Pinecone/sec-10k-qa](https://huggingface.co/datasets/Tim-Pinecone/sec-10k-qa). To rebuild from scratch:

```bash
# Build and run the dataset builder
docker build -f Dockerfile.dataset -t sec-dataset-builder .
docker run -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
  sec-dataset-builder all
```

Or deploy to EC2 via `deploy.sh` in the old dataset-builder infrastructure.

---

## How Embedding Persistence Works

Embeddings are expensive (OpenAI API charges per token). The benchmark saves them:

1. **Phase 1 — Embed upfront:** Questions are embedded once and saved to `s3://<bucket>/sec-benchmark/<run>/embeddings/questions_<model>.parquet`. Each chunker config's chunks are embedded and saved to `chunks_<Chunker>_<size>_<model>.parquet`.

2. **Phase 2 — Evaluate from cache:** All evaluation uses the saved parquets. Re-running the same config skips embedding entirely (checkpoint resume).

3. **Share via HuggingFace:** Run `upload_embeddings_to_hf.py` to publish embeddings publicly so anyone can run evaluations without API keys.

Parquet schema:
```
# Questions
question_id: int32
question:    large_string
embedding:   list<float32>[1536]

# Chunks
doc_id:      int32
chunk_idx:   int32
chunk_text:  large_string
embedding:   list<float32>[1536]
```
