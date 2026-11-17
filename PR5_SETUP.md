# PR5 — OpenSearch HNSW + BM25/dense/hybrid search

**Requires** Stage 1–2 + PR3 + PR4. PR4 must have produced `manifest.json`, `vectors.npy`, and `paper_ids.json` under S3. PR5 **does not regenerate paper embeddings**.

PR5 adds `scholarsearch/vector_index.py`, `scholarsearch/query_embed.py`, and `scholarsearch/search.py`. It makes a **separate** HNSW index (`scholarsearch-malware-hnsw-v1`) with both BM25 metadata and saved 768-d SPECTER2 vectors. The original PR3 BM25 index is left alone.

## 1. Check your local settings

Your existing `.env` needs at least:

```dotenv
AWS_PROFILE=scholarsearch-terraform
AWS_REGION=us-west-2
SCHOLARSEARCH_S3_ROOT=s3://scholarsearch-1a1f8b35f8/scholarsearch
SCHOLARSEARCH_QUERY="malware detection"
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_HYBRID_INDEX=scholarsearch-malware-hnsw-v1
HF_TOKEN=hf_REPLACE_WITH_YOUR_READ_ONLY_TOKEN
```

Keep `.env` Git-ignored. `HF_TOKEN` is only required when Hugging Face needs authentication. Do not commit it. The main project `.venv` has `opensearch-py`, `numpy`, `polars`, `boto3`. The **separate** `.venv-embed` has `adapters==1.3.0`, `transformers==4.57.6`, and `torch` to encode search queries. Do not install `adapters` into the main environment if it requires Transformers 5.x.

## 2. Run offline tests (NO EC2)

```bash
uv run --env-file .env python -m unittest discover -s tests -p 'test_pr5.py' -v
```

## 3. Start temporary OpenSearch

Your previous EC2 was destroyed, so recreate it using the **existing** Terraform module:

```bash
cd infra/aws/ephemeral
AWS_PROFILE=scholarsearch-terraform terraform init
AWS_PROFILE=scholarsearch-terraform terraform apply
terraform output -raw instance_id
```

Wait for the instance to register in SSM and for the OpenSearch container to start. From **this same directory**, run in a dedicated terminal:

```bash
aws ssm start-session \
  --target "$(terraform output -raw instance_id)" \
  --region us-west-2 --profile scholarsearch \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["9200"],"localPortNumber":["9200"]}'
```

Leave the tunnel running. In another terminal, from repository root:

```bash
curl http://localhost:9200
```

OpenSearch 2.15 supports Lucene HNSW with `innerproduct` for L2-normalized vectors. The mapping uses one shard and zero replicas for a single-node disposable instance.

## 4. Create the HNSW index (main .venv, not .venv-embed)

```bash
cd ~/search_projects/search-algorithms
uv run --env-file .env python -m scholarsearch.vector_index
```

Expected: `indexed: 300`, `count: 300`, `dimension: 768`. This command requires a **fresh index**; if the destination index already exists, it will refuse to overwrite it. Only run the following if you intentionally want to **delete and rebuild** that HNSW index:

```bash
uv run --env-file .env python -m scholarsearch.vector_index --recreate
```

It verifies the source digest, model/adapter, stored ID ordering, dtype, dimensionality, and unit normalization before indexing. On reapplying ephemeral Terraform from scratch there is no existing index.

## 5. Encode one search query (separate .venv-embed)

```bash
set -a; source .env; set +a
.venv-embed/bin/python -m scholarsearch.query_embed \
  --text 'graph neural network malware detection' \
  --output /tmp/scholarsearch-query.json \
  --device cpu
```

Query encoder = `allenai/specter2_base` + `allenai/specter2_adhoc_query`. Paper encoder in PR4 = same base + `allenai/specter2` proximity adapter. These are deliberately different. `/tmp/scholarsearch-query.json` is a **local** vector file; not uploaded to S3.

## 6. Compare search modes (main .venv)

```bash
uv run --env-file .env python -m scholarsearch.search \
  --query 'graph neural network malware detection' --mode bm25

uv run --env-file .env python -m scholarsearch.search \
  --query 'graph neural network malware detection' --mode dense \
  --query-vector-file /tmp/scholarsearch-query.json

uv run --env-file .env python -m scholarsearch.search \
  --query 'graph neural network malware detection' --mode hybrid \
  --query-vector-file /tmp/scholarsearch-query.json
```

Hybrid retrieves top `candidate_k` from **each** side (default 50), then applies reciprocal rank fusion with `1/(60+rank)`. Scores are ranking-fusion values, not BM25 or cosine similarities. Output excludes the full embedding arrays. For a different query, generate a **new** query-vector file or the search command will reject the mismatch.

## 7. Shut down cloud resources

Once testing is complete, stop the tunnel and run:

```bash
cd ~/search_projects/search-algorithms/infra/aws/ephemeral
AWS_PROFILE=scholarsearch-terraform terraform destroy
```

Only destroys the disposable EC2/OpenSearch resources under the ephemeral Terraform state; persistent S3 remains. The HNSW index is rebuilt from S3 next session. Keep the `.env` secret, and don't leave EC2 running after testing.

## Scope / limitations

- Smoke tests are offline fakes. A full OpenSearch 2.15 + SSM integration run must happen on your AWS instance.
- PR5 does not measure relevance improvements or ANN recall. That's a later evaluated benchmark with judged queries and a larger corpus.
- For this 300-paper prototype, query encoding runs locally once per query and is passed as JSON into the main search process. A future API/worker can keep the encoder warm to avoid repeated model startup.
