# This state is intentionally independent of ephemeral compute.
# Keep this directory/state after destroying infra/aws/ephemeral.
resource "random_id" "bucket_suffix" { byte_length = 5 }

resource "aws_s3_bucket" "research" {
  bucket = "${var.project}-${random_id.bucket_suffix.hex}"
  tags   = { Project = var.project, Persistence = "persistent" }
  lifecycle { prevent_destroy = true }
}

resource "aws_s3_bucket_public_access_block" "research" {
  bucket                  = aws_s3_bucket.research.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "research" {
  bucket = aws_s3_bucket.research.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}
