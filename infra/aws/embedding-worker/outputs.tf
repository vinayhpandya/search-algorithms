output "instance_id" {
  value = aws_instance.worker.id
}
output "source_uri" {
  value = "s3://${var.bucket_name}/${local.source_key}"
}
output "experiment_root" {
  value = local.s3_root
}
output "region" {
  value = var.aws_region
}
