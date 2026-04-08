output "instance_id" {
  value = aws_instance.benchmark.id
}

output "ami_name" {
  value = data.aws_ami.deep_learning.name
}

output "s3_results_path" {
  value = "s3://${var.s3_bucket}/sec-benchmark/results/"
}

output "watch_log_command" {
  value = "aws s3 cp s3://${var.s3_bucket}/sec-benchmark/logs/benchmark-run.log -"
}

output "ssm_session_command" {
  value = "aws ssm start-session --target ${aws_instance.benchmark.id} --region ${var.region}"
}

output "ecr_image" {
  value = var.ecr_image
}
