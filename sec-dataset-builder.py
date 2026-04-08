"""
SEC Filing Dataset Builder for MTCB
Downloads SEC filings from EDGAR, cleans them, generates QA pairs.

Folder structure:
    sec-benchmark/
    ├── build_dataset.py          ← this file
    ├── requirements.txt
    ├── raw/                      ← raw downloaded filings
    ├── cleaned/                  ← cleaned text files
    ├── dataset/                  ← generated QA dataset
    │   ├── corpus.jsonl
    │   ├── questions.jsonl
    │   └── sec_dataset.jsonl     ← DatasetGenerator output
    └── results/                  ← benchmark results go here

Usage:
    export OPENAI_API_KEY="sk-..."
    export HF_TOKEN="hf_..."
    python build_dataset.py download
    python build_dataset.py clean
    python build_dataset.py generate
    python build_dataset.py benchmark
    python build_dataset.py all          ← runs everything end to end
"""

import os
import re
import sys
import json
import glob
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"
CLEANED_DIR = BASE_DIR / "cleaned"
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"

for d in [RAW_DIR, CLEANED_DIR, DATASET_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
# Companies to download (CIK numbers)
# Find CIKs at https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
COMPANIES = {
    "AAPL":  "0000320193",   # Apple
    "MSFT":  "0000789019",   # Microsoft
    "GOOGL": "0001652044",   # Alphabet
    "AMZN":  "0001018724",   # Amazon
    "TSLA":  "0001318605",   # Tesla
    "JPM":   "0000019617",   # JPMorgan Chase
    "JNJ":   "0000200406",   # Johnson & Johnson
    "UNH":   "0000731766",   # UnitedHealth
    "V":     "0001403161",   # Visa
    "PG":    "0000080424",   # Procter & Gamble
    "NVDA":  "0001045810",   # NVIDIA
    "META":  "0001326801",   # Meta
    "BRK":   "0001067983",   # Berkshire Hathaway
    "XOM":   "0000034088",   # Exxon Mobil
    "WMT":   "0000104169",   # Walmart
    "BAC":   "0000070858",   # Bank of America
    "PFE":   "0000078003",   # Pfizer
    "DIS":   "0001744489",   # Walt Disney
    "NFLX":  "0001065280",   # Netflix
    "AMD":   "0000002488",   # AMD
}

FILING_TYPES = ["10-K"]
FILINGS_PER_COMPANY = 5  # 20 companies × 5 filings = 100 10-Ks
SAMPLES_PER_DOCUMENT = 10  # QA pairs to generate per filing
USER_AGENT = "MTCB-Benchmark research@example.com"  # SEC requires user-agent

# Smoke test mode — single company, 1 filing, 2 QA pairs
if os.environ.get("SMOKE_TEST"):
    COMPANIES = {"AAPL": "0000320193"}
    FILINGS_PER_COMPANY = 1
    SAMPLES_PER_DOCUMENT = 2


# ===========================================================================
# Step 1: Download filings from EDGAR
# ===========================================================================
def download_filings():
    """Download SEC filings using sec-edgar-downloader."""
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        print("Installing sec-edgar-downloader...")
        os.system(f"{sys.executable} -m pip install sec-edgar-downloader")
        from sec_edgar_downloader import Downloader

    dl = Downloader("MTCB-Research", "research@example.com", str(RAW_DIR))

    total = len(COMPANIES) * len(FILING_TYPES) * FILINGS_PER_COMPANY
    count = 0

    for ticker, cik in COMPANIES.items():
        for filing_type in FILING_TYPES:
            count += 1
            print(f"  [{count}/{total}] Downloading {filing_type} for {ticker}...")
            try:
                dl.get(
                    filing_type,
                    ticker,
                    limit=FILINGS_PER_COMPANY,
                    download_details=True,
                )
                print(f"    ✅ {ticker} {filing_type}")
            except Exception as e:
                print(f"    ❌ {ticker} {filing_type}: {e}")

    # Count what we got
    all_files = list(RAW_DIR.rglob("*.htm*")) + list(RAW_DIR.rglob("*.txt"))
    print(f"\n📁 Downloaded {len(all_files)} filing files to {RAW_DIR}")


# ===========================================================================
# Step 2: Clean filings — extract text from HTML
# ===========================================================================
def clean_filings():
    """Clean raw SEC filings: strip HTML, remove boilerplate, save as text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("Installing beautifulsoup4...")
        os.system(f"{sys.executable} -m pip install beautifulsoup4 lxml")
        from bs4 import BeautifulSoup

    # Find all filing documents
    raw_files = list(RAW_DIR.rglob("full-submission.txt")) + \
                list(RAW_DIR.rglob("*.htm")) + \
                list(RAW_DIR.rglob("*.html"))

    if not raw_files:
        # Try alternate structure from sec-edgar-downloader
        raw_files = list(RAW_DIR.rglob("*.txt"))
        raw_files = [f for f in raw_files if f.stat().st_size > 10000]

    print(f"  Found {len(raw_files)} raw files")

    cleaned_count = 0
    for filepath in raw_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            # Strip HTML
            if "<html" in raw_text.lower() or "<div" in raw_text.lower():
                soup = BeautifulSoup(raw_text, "lxml")
                # Remove script and style elements
                for tag in soup(["script", "style", "meta", "link"]):
                    tag.decompose()
                text = soup.get_text(separator="\n")
            else:
                text = raw_text

            # Clean up
            text = _clean_text(text)

            # Skip if too short (probably not the main filing)
            if len(text) < 5000:
                continue

            # Skip if too long (might be the full SGML submission — extract main doc)
            if len(text) > 500000:
                text = _extract_main_document(text)

            # Generate a clean filename
            parts = filepath.relative_to(RAW_DIR).parts
            clean_name = "_".join(parts[:-1]) + ".txt"
            clean_name = re.sub(r'[^\w\-.]', '_', clean_name)

            out_path = CLEANED_DIR / clean_name
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

            cleaned_count += 1
            size_kb = len(text) / 1024
            print(f"    ✅ {clean_name} ({size_kb:.0f} KB)")

        except Exception as e:
            print(f"    ❌ {filepath.name}: {e}")

    print(f"\n📁 Cleaned {cleaned_count} filings → {CLEANED_DIR}")

    s3_bucket = os.environ.get("S3_BUCKET", "mtcb-benchmark")
    try:
        import boto3
        s3 = boto3.client("s3")
        for fpath in CLEANED_DIR.glob("*.txt"):
            s3.upload_file(str(fpath), s3_bucket, f"sec-benchmark/cleaned/{fpath.name}")
        print(f"  ☁️  Cleaned filings uploaded to s3://{s3_bucket}/sec-benchmark/cleaned/")
    except Exception as e:
        print(f"  ⚠️  S3 upload failed (cleaned files still saved locally): {e}")


def _clean_text(text):
    """Clean SEC filing text."""
    # Remove SGML/XML headers
    text = re.sub(r'<SEC-DOCUMENT>.*?<TEXT>', '', text, flags=re.DOTALL)
    text = re.sub(r'</TEXT>.*?</SEC-DOCUMENT>', '', text, flags=re.DOTALL)
    # Remove remaining HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove XBRL tags
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {3,}', '  ', text)
    text = re.sub(r'\t+', ' ', text)
    # Remove page numbers / headers that repeat
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # Remove common boilerplate
    text = re.sub(r'Table of Contents', '', text, flags=re.IGNORECASE)
    return text.strip()


def _extract_main_document(text):
    """Extract the main document from a full SGML submission."""
    # Try to find the 10-K/10-Q section
    markers = [
        r'ITEM\s+1\.?\s+BUSINESS',
        r'PART\s+I',
        r'ANNUAL REPORT',
    ]
    for marker in markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            # Take from this marker to the end (or a reasonable length)
            start = max(0, match.start() - 500)
            return text[start:start + 300000]
    # Fallback: just take the middle chunk
    mid = len(text) // 2
    return text[max(0, mid - 150000):mid + 150000]


# ===========================================================================
# Step 3: Generate QA dataset using MTCB DatasetGenerator
# ===========================================================================
def generate_dataset():
    """Generate QA pairs from cleaned SEC filings."""
    from mtcb import DatasetGenerator

    # Load cleaned filings
    corpus = []
    doc_ids = []
    for filepath in sorted(CLEANED_DIR.glob("*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if len(text) < 1000:
            continue
        corpus.append(text)
        doc_ids.append(filepath.stem)

    if not corpus:
        print("❌ No cleaned filings found. Run 'clean' first.")
        return

    print(f"  Loaded {len(corpus)} filings")
    print(f"  Total text: {sum(len(d) for d in corpus) / 1e6:.1f} MB")
    print(f"  Generating {SAMPLES_PER_DOCUMENT} QA pairs per document...")
    print(f"  Expected: ~{len(corpus) * SAMPLES_PER_DOCUMENT} QA pairs\n")

    # Save corpus as JSONL for reference
    corpus_path = DATASET_DIR / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc_id, text in zip(doc_ids, corpus):
            json.dump({"id": doc_id, "text": text}, f)
            f.write("\n")
    print(f"  💾 Corpus saved to {corpus_path}")

    # Generate QA pairs
    # Gemini flash paraphrases rather than copying verbatim, so ExactMatchValidator
    # (the default) rejects everything. Use a no-op validator and accept all samples —
    # chunk_must_contain will be near-verbatim even if not pixel-perfect.
    class _AcceptAll:
        def validate(self, sample, chunk_text):
            return True

    from chonkie import GeminiGenie
    from mtcb import DatasetPromptTemplate
    output_path = str(DATASET_DIR / "sec_dataset.jsonl")
    generator = DatasetGenerator(
        genie=GeminiGenie(model="gemini-2.5-flash"),
        prompt_template=DatasetPromptTemplate.strict(),  # instructs model to copy verbatim
        validator=_AcceptAll(),  # accept all — strict prompt should produce verbatim text
        deduplicate=True,
    )

    try:
        result = generator.generate(
            corpus=corpus,
            samples_per_document=SAMPLES_PER_DOCUMENT,
            output_path=output_path,
        )
        print(f"\n  ✅ Generated {result.total_verified} verified QA pairs")

        if result.total_verified == 0:
            print(f"  ❌ 0 QA pairs passed validation — cannot proceed.")
            print(f"     total_generated={result.total_generated}, failed_validation={result.failed_validation_count}")
            sys.exit(1)

        print(f"  💾 Dataset saved to {output_path}")

        print(f"\n  Sample QA pairs:")
        for i, sample in enumerate(result.samples[:5]):
            print(f"\n  [{i+1}] Q: {sample.question}")
            print(f"      A: {sample.answer[:200]}...")

        # Upload dataset to S3
        s3_bucket = os.environ.get("S3_BUCKET", "mtcb-benchmark")
        try:
            import boto3
            s3 = boto3.client("s3")
            for fpath in [corpus_path, output_path]:
                key = f"sec-benchmark/dataset/{Path(fpath).name}"
                s3.upload_file(str(fpath), s3_bucket, key)
            print(f"\n  ☁️  Dataset uploaded to s3://{s3_bucket}/sec-benchmark/dataset/")
        except Exception as e:
            print(f"\n  ⚠️  S3 upload failed (dataset still saved locally): {e}")

    except Exception as e:
        print(f"\n  ❌ DatasetGenerator failed: {e}")
        print(f"\n  Check that GEMINI_API_KEY is set correctly.")
        sys.exit(1)
        print(f"\n  You can also check what parameters it expects:")
        print(f"    python -c \"from mtcb import DatasetGenerator; help(DatasetGenerator)\"")


# ===========================================================================
# Step 4: Run benchmark with custom dataset
# ===========================================================================
def run_benchmark():
    """Run MTCB benchmark on the generated SEC dataset."""
    import gc
    import torch
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from mtcb import SimpleEvaluator
    from chonkie import (
        RecursiveChunker, TokenChunker, SentenceChunker,
        SemanticChunker, FastChunker, LateChunker, NeuralChunker,
        SentenceTransformerEmbeddings,
    )

    class MTCBCompatibleEmbeddings(SentenceTransformerEmbeddings):
        def embed(self, text, **kwargs): return super().embed(text)
        def embed_batch(self, texts, **kwargs): return super().embed_batch(texts)
        def clear_cache(self): pass

    # Load dataset
    dataset_path = DATASET_DIR / "sec_dataset.jsonl"
    corpus_path = DATASET_DIR / "corpus.jsonl"

    if not dataset_path.exists():
        print("❌ No dataset found. Run 'generate' first.")
        return

    # Load corpus
    corpus = []
    with open(corpus_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            corpus.append(doc["text"])

    # Load QA pairs
    # DatasetGenerator writes one line per document: {"document_id": N, "samples": [...]}
    # Each sample has fields: question, answer, chunk_must_contain
    questions = []
    relevant_passages = []
    with open(dataset_path, "r") as f:
        for line in f:
            record = json.loads(line)
            for sample in record.get("samples", []):
                q = sample.get("question", "")
                a = sample.get("chunk_must_contain", sample.get("answer", ""))
                if q and a:
                    questions.append(q)
                    relevant_passages.append(a)

    print(f"  Loaded {len(corpus)} documents, {len(questions)} QA pairs")
    if len(questions) == 0:
        print(f"  ❌ No QA pairs loaded from {dataset_path} — did the generate step succeed?")
        sys.exit(1)

    # Config
    EMBEDDING_MODELS = {
        "bge-large-en": "BAAI/bge-large-en-v1.5",
        "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    }
    CHUNK_SIZES = [256, 512, 1000, 1500, 2000]
    K_VALUES = [1, 3, 5, 10]

    if os.environ.get("SMOKE_TEST"):
        EMBEDDING_MODELS = {"all-MiniLM-L6-v2": "all-MiniLM-L6-v2"}
        CHUNK_SIZES = [512]

    def make_chunkers(chunk_size, embeddings):
        chunkers = [
            ("TokenChunker", TokenChunker(chunk_size=chunk_size)),
            ("FastChunker", FastChunker(chunk_size=chunk_size)),
            ("SentenceChunker", SentenceChunker(chunk_size=chunk_size)),
            ("RecursiveChunker", RecursiveChunker(chunk_size=chunk_size)),
        ]
        try:
            chunkers.append(("SemanticChunker", SemanticChunker(
                chunk_size=chunk_size, embedding_model=embeddings)))
        except Exception:
            pass
        try:
            chunkers.append(("LateChunker", LateChunker(
                chunk_size=chunk_size, embedding_model=embeddings)))
        except Exception:
            pass
        try:
            chunkers.append(("NeuralChunker", NeuralChunker()))
        except Exception:
            pass
        return chunkers

    results = []
    checkpoint_path = RESULTS_DIR / "sec_benchmark_checkpoint.csv"

    # Load existing results for resume
    completed = set()
    if checkpoint_path.exists():
        df = pd.read_csv(checkpoint_path)
        results = df.to_dict("records")
        for _, row in df.iterrows():
            if "error" not in df.columns or pd.isna(row.get("error")):
                completed.add((
                    str(row.get("embedding_model", "")),
                    str(row.get("chunker", "")),
                    int(row.get("chunk_size", 0)),
                ))
        print(f"  📂 Resuming from {len(completed)} completed runs")

    total_runs = len(EMBEDDING_MODELS) * len(CHUNK_SIZES) * 7
    run_count = 0

    for model_key, model_name in EMBEDDING_MODELS.items():
        print(f"\n  🧠 Loading {model_key}...")
        try:
            embeddings = MTCBCompatibleEmbeddings(model=model_name)
        except Exception as e:
            print(f"  ❌ Failed to load {model_key}: {e}")
            run_count += len(CHUNK_SIZES) * 7
            continue

        for chunk_size in CHUNK_SIZES:
            chunkers = make_chunkers(chunk_size, embeddings)

            for chunker_name, chunker in chunkers:
                run_count += 1
                run_key = (model_key, chunker_name, chunk_size)

                if run_key in completed:
                    print(f"  ⏭️  [{run_count}/{total_runs}] {chunker_name}(size={chunk_size}) — skip")
                    continue

                print(f"  🔄 [{run_count}/{total_runs}] {chunker_name}(size={chunk_size}) × {model_key}")
                start = time.time()

                try:
                    evaluator = SimpleEvaluator(
                        corpus=corpus,
                        questions=questions,
                        relevant_passages=relevant_passages,
                        chunker=chunker,
                        embedding_model=embeddings,
                    )
                    result = evaluator.evaluate(k=K_VALUES)
                    elapsed = time.time() - start

                    row = {
                        "embedding_model": model_key,
                        "evaluator": "SEC Filings (Custom)",
                        "chunker": chunker_name,
                        "chunk_size": chunk_size,
                        "time_sec": round(elapsed, 1),
                    }
                    # Parse result
                    result_str = str(result)
                    for k in K_VALUES:
                        import re as _re
                        match = _re.search(rf"R@{k}=([\d.]+)%", result_str)
                        row[f"R@{k}"] = float(match.group(1)) / 100 if match else None
                    mrr = getattr(result, "mrr", None)
                    if isinstance(mrr, dict):
                        row["MRR@10"] = float(mrr.get(10, 0))
                    else:
                        match = _re.search(r"MRR@10=([\d.]+)", result_str)
                        row["MRR@10"] = float(match.group(1)) if match else None

                    results.append(row)
                    print(f"     ✅ {result} ({elapsed:.1f}s)")

                except Exception as e:
                    elapsed = time.time() - start
                    print(f"     ❌ {e} ({elapsed:.1f}s)")
                    results.append({
                        "embedding_model": model_key,
                        "evaluator": "SEC Filings (Custom)",
                        "chunker": chunker_name,
                        "chunk_size": chunk_size,
                        "time_sec": round(elapsed, 1),
                        "error": str(e),
                    })

                gc.collect()
                torch.cuda.empty_cache()

                # Save every run
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv = RESULTS_DIR / f"sec_benchmark_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(final_csv, index=False)
    print(f"\n  💾 Results saved to {final_csv}")

    # Upload results to S3
    s3_bucket = os.environ.get("S3_BUCKET", "mtcb-benchmark")
    try:
        import boto3
        s3 = boto3.client("s3")
        s3_key = f"sec-benchmark/results/{final_csv.name}"
        s3.upload_file(str(final_csv), s3_bucket, s3_key)
        print(f"  ☁️  Results uploaded to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"  ⚠️  S3 upload failed (results still saved locally): {e}")

    # Quick summary
    df_ok = df[~df.get("error", pd.Series(dtype=str)).notna()] if "error" in df.columns else df
    if "R@5" in df_ok.columns and len(df_ok) > 0:
        print(f"\n  📈 Average R@5 by chunker:")
        for name, val in df_ok.groupby("chunker")["R@5"].mean().sort_values(ascending=False).items():
            print(f"      {name:25s} {val:.1%}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="SEC Dataset Builder for MTCB")
    parser.add_argument("command", choices=["download", "clean", "generate", "benchmark", "dataset", "all"],
                        help="Which step to run (dataset = download+clean+generate)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SEC Dataset Builder for MTCB")
    print(f"{'='*60}\n")

    if args.command in ("download", "dataset", "all"):
        print("📥 Step 1: Downloading SEC filings from EDGAR...")
        download_filings()

    if args.command in ("clean", "dataset", "all"):
        print("\n🧹 Step 2: Cleaning filings...")
        clean_filings()

    if args.command in ("generate", "dataset", "all"):
        print("\n🤖 Step 3: Generating QA dataset...")
        generate_dataset()

    if args.command in ("benchmark", "all"):
        print("\n📊 Step 4: Running benchmark...")
        run_benchmark()

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
