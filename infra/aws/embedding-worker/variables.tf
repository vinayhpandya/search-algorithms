variable "aws_region" {
  description = "Same region as ScholarSearch S3 and prior experiments"
  type        = string
  default     = "us-west-2"
}
variable "instance_type" {
  description = "CPU-only instance; verify eligibility and pricing in your AWS account before applying"
  type        = string
  default     = "m7i-flex.large"
}
variable "root_volume_gb" {
  type    = number
  default = 40
  validation {
    condition     = var.root_volume_gb >= 30
    error_message = "Root volume must be at least 30 GiB for model weights and packages."
  }
}
variable "bucket_name" {
  type    = string
  default = "scholarsearch-1a1f8b35f8"
}
variable "experiment_prefix" {
  description = "S3 key prefix, not an s3:// URI; source code and checkpoints both live here"
  type        = string
  default     = "scholarsearch/experiments/experiment-b"
}
variable "max_runtime_hours" {
  type    = number
  default = 12
  validation {
    condition     = var.max_runtime_hours >= 1 && var.max_runtime_hours <= 24 && floor(var.max_runtime_hours) == var.max_runtime_hours
    error_message = "Use a whole number of hours from 1 to 24."
  }
}
