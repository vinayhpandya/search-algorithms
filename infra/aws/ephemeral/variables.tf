variable "aws_region" {
  type    = string
  default = "us-west-2"
}
variable "instance_type" {
  description = "8 GiB is recommended for a single-node OpenSearch experiment. Check Free-plan EC2 eligibility and quotas before apply."
  type        = string
  default     = "m7i-flex.large"
}
variable "root_volume_gb" {
  type    = number
  default = 40
  validation {
    condition     = var.root_volume_gb >= 30
    error_message = "Allow at least 30 GiB for the OS, Docker image and test indexes."
  }
}
variable "opensearch_version" {
  description = "Pin to the OpenSearch version used by the repo; do not use latest."
  type        = string
  default     = "2.15.0"
}
