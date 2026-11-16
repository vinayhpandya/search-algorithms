# ScholarSearch AWS infrastructure (experimental)

This is an **additive** Terraform scaffold for the existing research repo.
It creates **real billable AWS resources** when you run `terraform apply`.
**Neither configuration has been applied to an AWS account or integration-tested.**

## Cost and account safety

- Verify the AWS account's Free-plan service restrictions, EC2 quota, remaining promotional credits, and selected `us-west-2` region **before** `terraform apply`.
- Create AWS Budgets alerts first. An alert does not stop resources automatically.
- `t3.large` (8 GiB) may not be eligible under your Free plan; a `terraform apply` may be denied. Do not assume that free credits authorize all EC2 instance types.
- Approximate example (NOT a price quote): 30 hours of `t3.large` at $0.0832/hr = $2.50 compute; a 40-GiB gp3 disk at $0.08/GB-month = $3.20/month while retained; public IPv4 at $0.005/hr ≈ $0.15 for 30 hours; plus S3, transfer, requests, CPU-credit overages and taxes. Rates vary by region and date.
- **No NAT Gateway** and **no inbound security-group rules**. Instance uses a public IP only for outbound AWS/Docker access and SSM connections.
- Disables the OpenSearch security plugin for a disposable, *localhost-bound* evaluation instance. Do not expose or reuse this config for production.
- Treat all OpenSearch indexes on the instance as **disposable**. `terraform destroy` deletes the EC2 root disk and its indexes. Save source data/embeddings elsewhere first.

## Prerequisites on your own Mac

- Terraform installed and `terraform version` working.
- AWS CLI v2.32+ with `aws login --profile scholarsearch` as a **non-root IAM identity** with relevant EC2/IAM/S3/SSM permissions.
- `aws sts get-caller-identity --profile scholarsearch` succeeds (do not post account identifiers).
- AWS Session Manager Plugin available to the CLI; check `session-manager-plugin --version` and install from AWS's official docs if missing.
- The default VPC and default subnets exist in `us-west-2`. If your account has no default VPC, this module will fail intentionally rather than silently configuring a public network.

Set credentials for both Terraform modules:

```bash
export AWS_PROFILE=scholarsearch
export AWS_REGION=us-west-2
```

## Create S3 persistent data (once)

```bash
cd infra/aws/persistent
terraform init
terraform fmt -check
terraform validate
terraform plan -out=tfplan
terraform apply tfplan
terraform output -raw dataset_root
```

This uses a separate local Terraform state. The S3 bucket has `prevent_destroy = true`. To preserve access and recreate infrastructure reliably, **back up the state file** securely, but do not commit it. The `terraform.tfstate` file can contain sensitive information. Local state is for a personal prototype, not team collaboration.

Store data under e.g. `s3://BUCKET/scholarsearch/raw/`, `curated/`, `embeddings/`, `benchmarks/`. Stage 1's `--root s3://BUCKET/scholarsearch` uses these credentials; ensure IAM grants S3 access.

## Launch disposable OpenSearch (only when actively experimenting)

```bash
cd ../ephemeral
terraform init
terraform fmt -check
terraform validate
terraform plan -out=tfplan
terraform apply tfplan
terraform output -raw instance_id
```

**Wait until the EC2 instance is registered in Systems Manager**; `terraform apply` completing does not mean the Docker image has finished downloading or OpenSearch has started. Inspect `/var/log/cloud-init-output.log` through SSM if needed.

Install the Session Manager plugin if necessary. In a **separate terminal**, run:

```bash
aws ssm start-session \
  --target "$(terraform output -raw instance_id)" \
  --region us-west-2 --profile scholarsearch \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["9200"],"localPortNumber":["9200"]}'
```

Keep that terminal session open. In another terminal run `curl http://127.0.0.1:9200` to check the OpenSearch server. No SSH key or port 9200 ingress is required. The root disk holds OpenSearch data and is destroyed with the VM.

## Destroy only temporary resources

```bash
cd infra/aws/ephemeral # from repository root
terraform plan -destroy
terraform destroy
```

Be certain you are in `ephemeral`, not `persistent`. Persistent state is separate; `prevent_destroy` additionally protects the dataset bucket from accidental Terraform deletion (unless you deliberately remove that protection and empty the bucket). Stop running instances even when not in use; destroy after experiment sessions to stop compute and disk costs. S3 storage and some AWS charges continue.

## Integration still to implement

- Index the curated paper Parquet + SPECTER2 vector outputs into your existing OpenSearch mapping. The Stage 1/2 patch creates data and runs standalone Faiss benchmarks but does not bridge these automatically.
- OpenSearch IVF training via the Train Model API and index mapping with a trained model (version-specific). The benchmark's IVF is Faiss IVF-Flat.
- No GPU or Spark resources are created by this Terraform module.
- If later adding GCP/Azure, define separate providers/states; this module is AWS-specific.
