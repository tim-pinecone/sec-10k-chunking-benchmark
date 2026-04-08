variable "hf_token" {
  description = "HuggingFace token (set via TF_VAR_hf_token)"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "g4dn.xlarge"
}

variable "s3_bucket" {
  description = "S3 bucket for results and dataset cache"
  type        = string
  default     = "mtcb-benchmark"
}

variable "hf_token_ssm_path" {
  description = "SSM Parameter Store path for the HuggingFace token"
  type        = string
  default     = "/sec-benchmark/hf-token"
}

variable "ecr_image" {
  description = "Full ECR image URI, e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/sec-10k-benchmark:latest"
  type        = string
}

variable "key_name" {
  description = "EC2 key pair name for SSH access (optional)"
  type        = string
  default     = null
}

variable "openai_api_key" {
  description = "OpenAI API key (optional — only needed for OpenAI embedding models)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "benchmark_args" {
  description = "Extra args passed to benchmark.py, e.g. --embedding-model openai:text-embedding-3-large"
  type        = string
  default     = ""
}
