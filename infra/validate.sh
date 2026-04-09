#!/bin/bash
# ---------------------------------------------------------------------------
# validate.sh — local smoke test for the Jina v5 benchmark
#
# Builds Dockerfile.jina and runs jina_benchmark.py --dry-run inside it.
# Mirrors the EC2 environment exactly (linux/amd64, same image).
# No AWS credentials needed — S3_BUCKET="" disables all S3 ops; dataset
# loads from HuggingFace instead.
#
# Expected runtime: ~5-10 min (model download on first run, CPU inference)
# Subsequent runs are faster once the Docker layer cache is warm.
#
# Usage:
#   ./validate.sh            # full dry-run (5 docs, 25 questions)
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

echo ""
echo "=================================================="
echo "  validate.sh — Jina v5 benchmark smoke test"
echo "=================================================="
echo ""

# ---------------------------------------------------------------------------
# Docker check
# ---------------------------------------------------------------------------
echo "🐳 Checking Docker..."
docker info >/dev/null 2>&1 || {
  echo "❌ Docker is not running. Start Docker Desktop and try again."
  exit 1
}
echo "   ✅ Docker running"

# ---------------------------------------------------------------------------
# HF_TOKEN check (warn only — Jina v5 is public, but useful to have)
# ---------------------------------------------------------------------------
HF_TOKEN_VAL="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN_VAL" ] && [ -f "../.env" ]; then
  HF_TOKEN_VAL="$(grep '^HF_TOKEN=' ../.env 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
fi
if [ -z "$HF_TOKEN_VAL" ]; then
  echo "   ⚠️  HF_TOKEN not found — Jina v5 is public, proceeding without it"
fi

# ---------------------------------------------------------------------------
# Build — native arch on Apple Silicon to avoid slow x86 emulation
# EC2 uses linux/amd64 with CUDA; here we just need code + imports correct.
# ---------------------------------------------------------------------------
echo ""
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  echo "🏗️  Building Dockerfile.jina (native arm64, CPU torch — fast validation)..."
  BUILD_FLAGS="--platform linux/arm64 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu"
else
  echo "🏗️  Building Dockerfile.jina (linux/amd64, CUDA torch)..."
  BUILD_FLAGS="--platform linux/amd64"
fi
echo "   (first build downloads wheels — subsequent builds use cache)"

# shellcheck disable=SC2086
docker build $BUILD_FLAGS \
  -t sec-10k-benchmark-jina:validate \
  -f ../Dockerfile.jina \
  ..
echo "   ✅ Build complete"

# ---------------------------------------------------------------------------
# Run dry-run (1 doc, 5 questions — tests all code paths quickly)
# S3_BUCKET="" → _s3 = None in Python → all S3 ops skipped
# Dataset falls back to HuggingFace automatically
# ---------------------------------------------------------------------------
echo ""
echo "🧪 Running --dry-run (1 doc, 5 questions)..."
echo "   Tests: model load, RecursiveChunker, LateChunker, parquet write, eval"
echo "   (model download ~1.3GB on first run)"
echo ""

docker run --rm \
  -e S3_BUCKET="" \
  -e HF_TOKEN="${HF_TOKEN_VAL}" \
  sec-10k-benchmark-jina:validate \
  --dry-run --batch-size 4

echo ""
echo "=================================================="
echo "  ✅  validate.sh PASSED"
echo "=================================================="
