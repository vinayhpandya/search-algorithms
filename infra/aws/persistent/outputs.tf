output "bucket_name" { value = aws_s3_bucket.research.bucket }
output "dataset_root" { value = "s3://${aws_s3_bucket.research.bucket}/scholarsearch" }
