data "aws_vpc" "default" {
  default = true
}
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

# Copy of ../embedding-worker that runs scholarsearch.run_experiments instead of the
# embedding job. Same safety model: no inbound ports, S3 access limited to one prefix,
# SSM for inspection, and a watchdog. Differences: PyTorch only if install_ml, and the instance
# TERMINATES (not stops) when the job ends, so nothing keeps billing afterwards.
locals {
  source_key = "${var.experiment_prefix}/bootstrap/source.tar.gz"
  s3_root    = "s3://${var.bucket_name}/${var.experiment_prefix}"
  config_uri = "${local.s3_root}/configs/${var.config_name}.json"
  # SSM parameter ARNs omit the leading "/" of the parameter name.
  key_param_arn = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/${trimprefix(var.openai_key_parameter, "/")}"
}
data "aws_caller_identity" "current" {}

resource "aws_security_group" "worker" {
  name_prefix = "scholarsearch-exp-"
  description = "No inbound connections; outbound access for SSM, S3 and package installation"
  vpc_id      = data.aws_vpc.default.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Project = "scholarsearch", Purpose = "experiment-worker" }
}

resource "aws_iam_role" "worker" {
  name_prefix = "scholarsearch-experiment-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
resource "aws_iam_role_policy" "data" {
  name = "experiment-b-only"
  role = aws_iam_role.worker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadWriteExperimentObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "arn:aws:s3:::${var.bucket_name}/${var.experiment_prefix}/*"
      },
      {
        Sid      = "ListOnlyExperimentPrefix"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = "arn:aws:s3:::${var.bucket_name}"
        Condition = {
          StringLike = { "s3:prefix" = [var.experiment_prefix, "${var.experiment_prefix}/*"] }
        }
      }
    ]
  })
}

# Only created when openai_key_parameter is set: read ONE parameter, decrypt only via SSM.
resource "aws_iam_role_policy" "judge_key" {
  count = var.openai_key_parameter == "" ? 0 : 1
  name  = "read-judge-api-key"
  role  = aws_iam_role.worker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadOnlyTheJudgeApiKeyParameter"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = local.key_param_arn
      },
      {
        Sid       = "DecryptOnlyThroughSsm"
        Effect    = "Allow"
        Action    = ["kms:Decrypt"]
        Resource  = "*"
        Condition = { StringEquals = { "kms:ViaService" = "ssm.${var.aws_region}.amazonaws.com" } }
      }
    ]
  })
}
resource "aws_iam_instance_profile" "worker" {
  name_prefix = "scholarsearch-experiment-"
  role        = aws_iam_role.worker.name
}

resource "aws_instance" "worker" {
  ami                         = data.aws_ssm_parameter.al2023.value
  instance_type               = var.instance_type
  subnet_id                   = sort(data.aws_subnets.default.ids)[0]
  associate_public_ip_address = true
  vpc_security_group_ids      = [aws_security_group.worker.id]
  iam_instance_profile        = aws_iam_instance_profile.worker.name
  instance_initiated_shutdown_behavior = "terminate"
  metadata_options {
    http_tokens = "required"
  }
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }
  user_data = templatefile("${path.module}/bootstrap.sh.tftpl", {
    bucket            = var.bucket_name
    prefix            = var.experiment_prefix
    source_key        = local.source_key
    root              = local.s3_root
    config_uri        = local.config_uri
    config_name       = var.config_name
    install_ml        = var.install_ml
    key_parameter     = var.openai_key_parameter
    region            = var.aws_region
    runtime_hours     = var.max_runtime_hours
  })
  user_data_replace_on_change = true
  tags = {
    Name        = "scholarsearch-experiment-worker"
    Project     = "scholarsearch"
    Persistence = "ephemeral"
  }
  depends_on = [aws_iam_role_policy.data, aws_iam_role_policy.judge_key, aws_iam_role_policy_attachment.ssm]
}
