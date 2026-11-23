variable "aws_region" {
  description = "Same region as ScholarSearch S3 and prior experiments"
  type        = string
  default     = "us-west-2"
}
variable "instance_type" {
  description = "CPU-only instance; 8 GiB is enough at 15K, including rerankers. Verify pricing/eligibility first."
  type        = string
  default     = "m7i-flex.large"
}
variable "root_volume_gb" {
  type    = number
  default = 30
  validation {
    condition     = var.root_volume_gb >= 30
    error_message = "Amazon Linux 2023 AMIs require at least 30 GiB."
  }
}
variable "bucket_name" {
  type    = string
  default = "scholarsearch-1a1f8b35f8"
}
variable "experiment_prefix" {
  description = "S3 key prefix (not an s3:// URI). The worker can read/write only under this prefix."
  type        = string
  default     = "scholarsearch/experiments/experiment-b"
}
variable "config_name" {
  description = "Experiment config uploaded by scripts/package_source.sh to <prefix>/configs/<name>.json"
  type        = string
  default     = "baseline-15k"
  validation {
    condition     = can(regex("^[A-Za-z0-9_-]+$", var.config_name))
    error_message = "Use only letters, digits, '-' and '_'."
  }
}
variable "max_runtime_hours" {
  description = "Watchdog: the instance terminates after this many hours even if the job hangs"
  type        = number
  default     = 3
  validation {
    condition     = var.max_runtime_hours >= 1 && var.max_runtime_hours <= 24 && floor(var.max_runtime_hours) == var.max_runtime_hours
    error_message = "Use a whole number of hours from 1 to 24."
  }
}
variable "install_ml" {
  description = "Install CPU PyTorch, sentence-transformers and fastembed (needed for cross-encoder / ColBERT rerankers)"
  type        = bool
  default     = false
}
variable "openai_key_parameter" {
  description = "SSM SecureString parameter name holding OPENAI_API_KEY for the judge stage, e.g. /scholarsearch/openai-api-key. Empty = no key, no SSM permission."
  type        = string
  default     = ""
  validation {
    condition     = var.openai_key_parameter == "" || can(regex("^/[A-Za-z0-9_./-]+$", var.openai_key_parameter))
    error_message = "Use a parameter name starting with '/', e.g. /scholarsearch/openai-api-key."
  }
}
