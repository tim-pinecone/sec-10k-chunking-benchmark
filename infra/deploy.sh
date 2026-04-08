#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
ENV_FILE="../.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "❌  $ENV_FILE not found."
  exit 1
fi
set -a; source "$ENV_FILE"; set +a

# ---------------------------------------------------------------------------
# Credential check — fail fast before doing any work
# ---------------------------------------------------------------------------
echo "🔐 Checking AWS credentials..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
  echo "❌  AWS credentials expired or invalid."
  echo "    Run: aws sso login --profile ${AWS_PROFILE:-default}"
  exit 1
}
echo "   ✅  Account: ${ACCOUNT_ID}"

REGION="${AWS_REGION:-us-east-1}"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SKIP_BUILD=0
USE_JINA=0
for arg in "$@"; do
  [ "$arg" = "--skip-build" ] && SKIP_BUILD=1
  [ "$arg" = "--jina" ]       && USE_JINA=1
done

if [ "$USE_JINA" -eq 1 ]; then
  ECR_REPO="sec-10k-benchmark-jina"
  DOCKERFILE="../Dockerfile.jina"
  echo "   Mode: Jina v5 benchmark (jina_benchmark.py)"
else
  ECR_REPO="sec-10k-benchmark"
  DOCKERFILE="../Dockerfile"
  echo "   Mode: Standard benchmark (benchmark.py)"
fi

ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:latest"

# ---------------------------------------------------------------------------
# Ensure ECR repo exists
# ---------------------------------------------------------------------------
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" \
  --query 'repositories[0].repositoryUri' --output text 2>/dev/null || \
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}" \
  --query 'repository.repositoryUri' --output text

# ---------------------------------------------------------------------------
# Build + push Docker image (skipped with --skip-build)
# ---------------------------------------------------------------------------
if [ "$SKIP_BUILD" -eq 0 ]; then
  echo ""
  echo "🐳 Building Docker image..."
  docker build --platform linux/amd64 \
    -t "${ECR_REPO}:latest" \
    -f "${DOCKERFILE}" \
    ..
  echo "   ✅  Build complete"

  echo "📤 Pushing to ECR..."
  aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"
  docker tag "${ECR_REPO}:latest" "${ECR_IMAGE}"
  docker push "${ECR_IMAGE}"
  echo "   ✅  Pushed ${ECR_IMAGE}"
else
  echo "   ⏭️  Skipping Docker build (--skip-build)"
fi

# ---------------------------------------------------------------------------
# Upload dataset parquet files to S3 if not already there
# ---------------------------------------------------------------------------
echo ""
echo "📦 Checking dataset in S3..."
for split in corpus questions; do
  KEY="sec-benchmark/dataset/${split}.parquet"
  LOCAL="../dataset/${split}/train-00000-of-00001.parquet"
  aws s3api head-object --bucket "${S3_BUCKET:-mtcb-benchmark}" --key "$KEY" \
    --region "${REGION}" >/dev/null 2>&1 || {
    if [ -f "$LOCAL" ]; then
      aws s3 cp "$LOCAL" "s3://${S3_BUCKET:-mtcb-benchmark}/${KEY}"
      echo "   ✅  Uploaded ${split}.parquet"
    else
      echo "   ⚠️   ${split}.parquet not found locally — benchmark will load from HuggingFace"
    fi
  }
done

# ---------------------------------------------------------------------------
# Terraform
# ---------------------------------------------------------------------------
export TF_VAR_hf_token="${HF_TOKEN}"
export TF_VAR_region="${REGION}"
export TF_VAR_s3_bucket="${S3_BUCKET:-mtcb-benchmark}"
export TF_VAR_instance_type="${INSTANCE_TYPE:-g4dn.xlarge}"
export TF_VAR_ecr_image="${ECR_IMAGE}"
export TF_VAR_openai_api_key="${OPENAI_API_KEY:-}"
export TF_VAR_benchmark_args="${BENCHMARK_ARGS:-}"

if [ -n "${KEY_NAME:-}" ]; then
  export TF_VAR_key_name="${KEY_NAME}"
fi
if [ -n "${AWS_PROFILE:-}" ]; then
  export AWS_PROFILE
fi

terraform init -upgrade -input=false

# Strip --skip-build and --jina before passing remaining args to terraform
TERRAFORM_ARGS=()
for arg in "$@"; do
  [ "$arg" != "--skip-build" ] && [ "$arg" != "--jina" ] && TERRAFORM_ARGS+=("$arg")
done

SUBCMD="${TERRAFORM_ARGS[0]:-apply}"
REST=("${TERRAFORM_ARGS[@]:1}")

case "$SUBCMD" in
  apply|destroy)
    terraform "$SUBCMD" -input=false -auto-approve ${REST[@]+"${REST[@]}"}
    ;;
  *)
    terraform "$SUBCMD" ${REST[@]+"${REST[@]}"}
    ;;
esac
