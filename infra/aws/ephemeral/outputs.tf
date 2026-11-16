output "instance_id" { value = aws_instance.search.id }
output "region" { value = var.aws_region }
output "port_forward_command" {
  description = "Command to securely access OpenSearch through AWS SSM"

  value = "aws ssm start-session --target ${aws_instance.search.id} --region ${var.aws_region} --document-name AWS-StartPortForwardingSession --parameters '${jsonencode({ portNumber = ["9200"], localPortNumber = ["9200"] })}' --profile scholarsearch-terraform"
}
