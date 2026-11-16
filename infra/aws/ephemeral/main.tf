# Requires an existing DEFAULT VPC with a default/public subnet.
# No inbound rules: connect only through AWS Systems Manager Session Manager.
data "aws_vpc" "default" { default = true }
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "aws_security_group" "search" {
  name_prefix = "scholarsearch-experiment-"
  description = "No ingress; SSM via outbound HTTPS only"
  vpc_id      = data.aws_vpc.default.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Project = "scholarsearch", Persistence = "ephemeral" }
}

resource "aws_iam_role" "search" {
  name_prefix = "scholarsearch-ssm-"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.search.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
resource "aws_iam_instance_profile" "search" {
  name_prefix = "scholarsearch-profile-"
  role        = aws_iam_role.search.name
}

resource "aws_instance" "search" {
  ami                         = data.aws_ssm_parameter.al2023.value
  instance_type               = var.instance_type
  subnet_id                   = sort(data.aws_subnets.default.ids)[0]
  associate_public_ip_address = true # outbound access without a NAT gateway; no inbound allowed
  vpc_security_group_ids      = [aws_security_group.search.id]
  iam_instance_profile        = aws_iam_instance_profile.search.name
  user_data = templatefile("${path.module}/bootstrap.sh.tftpl", {
    opensearch_version = var.opensearch_version
  })
  user_data_replace_on_change = true
  metadata_options {
    http_tokens = "required"
  }
  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }
  tags = { Name = "scholarsearch-opensearch", Project = "scholarsearch", Persistence = "ephemeral" }
}
