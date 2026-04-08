terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "defaultForAz"
    values = ["true"]
  }
}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ---------------------------------------------------------------------------
# SSM — HuggingFace token
# ---------------------------------------------------------------------------

resource "aws_ssm_parameter" "hf_token" {
  name  = var.hf_token_ssm_path
  type  = "SecureString"
  value = var.hf_token
}

# ---------------------------------------------------------------------------
# IAM — S3 + SSM + ECR pull
# ---------------------------------------------------------------------------

resource "aws_iam_role" "benchmark" {
  name = "sec-10k-benchmark-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "benchmark" {
  name = "sec-10k-benchmark-policy"
  role = aws_iam_role.benchmark.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3Results"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*",
        ]
      },
      {
        Sid      = "SSMHFToken"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = "arn:aws:ssm:${var.region}:*:parameter${var.hf_token_ssm_path}"
      },
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
        ]
        Resource = "arn:aws:ecr:${var.region}:*:repository/sec-10k-benchmark"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.benchmark.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "benchmark" {
  name = "sec-10k-benchmark-profile"
  role = aws_iam_role.benchmark.name
}

# ---------------------------------------------------------------------------
# Security group — outbound-only
# ---------------------------------------------------------------------------

resource "aws_security_group" "benchmark" {
  name        = "sec-10k-benchmark-sg"
  description = "SEC 10K benchmark outbound only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# EC2 instance
# ---------------------------------------------------------------------------

locals {
  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    region            = var.region
    hf_token_ssm_path = var.hf_token_ssm_path
    s3_bucket         = var.s3_bucket
    ecr_image         = var.ecr_image
    openai_api_key    = var.openai_api_key
    benchmark_args    = var.benchmark_args
  })
}

resource "aws_instance" "benchmark" {
  ami                         = data.aws_ami.deep_learning.id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  iam_instance_profile        = aws_iam_instance_profile.benchmark.name
  associate_public_ip_address = true
  key_name                    = var.key_name

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 75
    delete_on_termination = true
  }

  user_data = local.user_data

  tags = {
    Name    = "sec-10k-benchmark"
    Project = "mtcb"
  }
}
