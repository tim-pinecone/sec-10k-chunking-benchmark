#!/bin/bash
set -euo pipefail
exec > /var/log/benchmark.log 2>&1

trap '
  echo "=== EXIT TRAP: uploading logs, shutting down ==="
  aws s3 cp /var/log/benchmark.log     s3://${s3_bucket}/sec-benchmark/logs/benchmark.log     2>/dev/null || true
  aws s3 cp /var/log/benchmark-run.log s3://${s3_bucket}/sec-benchmark/logs/benchmark-run.log 2>/dev/null || true
  shutdown -h now
' EXIT

echo "=== SEC 10-K Benchmark starting ==="
date

echo "Waiting for IAM credentials..."
until aws sts get-caller-identity >/dev/null 2>&1; do sleep 5; done
echo "IAM credentials ready."

# Fetch HF token from SSM before set -x so it never appears in logs
HF_TOKEN=$(aws ssm get-parameter \
  --region ${region} \
  --name "${hf_token_ssm_path}" \
  --with-decryption \
  --query Parameter.Value \
  --output text)

set -x

# Install Docker
dnf install -y docker
systemctl enable --now docker

# Install NVIDIA Container Toolkit so --gpus all works
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# ECR login — extract registry from image URI
ECR_REGISTRY=$(echo "${ecr_image}" | cut -d/ -f1)
aws ecr get-login-password --region ${region} | \
  docker login --username AWS --password-stdin "$ECR_REGISTRY"

# Pull pre-baked image
docker pull ${ecr_image}

# Run benchmark — volumes let checkpoint/dataset/cache survive container restarts
mkdir -p /data/results /data/cache /data/dataset
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -e OPENAI_API_KEY="${openai_api_key}" \
  -e S3_BUCKET="${s3_bucket}" \
  -e PYTHONUNBUFFERED=1 \
  -v /data/results:/app/results \
  -v /data/cache:/app/cache \
  -v /data/dataset:/app/dataset \
  ${ecr_image} ${benchmark_args} 2>&1 | tee /var/log/benchmark-run.log

echo "=== Benchmark complete ==="
date
