output "instance_id" {
  value = aws_instance.worker.id
}
output "config_uri" {
  value = local.config_uri
}
output "runs_uri" {
  description = "Each run writes a folder here"
  value       = "${local.s3_root}/runs/"
}
output "logs_uri" {
  value = "${local.s3_root}/worker-logs/"
}
output "region" {
  value = var.aws_region
}
