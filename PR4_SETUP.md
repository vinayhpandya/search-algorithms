# PR4 — SPECTER2 paper embeddings

**Requires the earlier Stage 1/2 and PR3 patches.** No EC2 or OpenSearch server is needed for this job.

## Dependencies

Install the adapter-capable model integration **in a separate environment if your existing `transformers` version conflicts**:

```bash
uv add adapters
```

The upstream SPECTER2 model uses `allenai/specter2_base` plus the
`allenai/specter2` proximity adapter. Python imports for the model are lazy;
unit tests do not download weights.

## Configure

Use the existing `.env` with:

```dotenv
AWS_PROFILE=scholarsearch-terraform
AWS_REGION=us-west-2
SCHOLARSEARCH_S3_ROOT=s3://YOUR_BUCKET/scholarsearch
SCHOLARSEARCH_QUERY="malware detection"
OPENALEX_API_KEY=YOUR_KEY
EMBEDDING_BATCH_SIZE=4
```

`.env` must be Git-ignored. `uv run --env-file .env` loads these variables
for Python, including boto3's AWS profile. No permanent AWS access keys needed.

## Commands

```bash
uv run --env-file .env python -m unittest discover -s tests -p 'test_embeddings.py' -v
uv run --env-file .env python -m scholarsearch.embeddings --help
uv run --env-file .env python -m scholarsearch.embeddings --batch-size 4
```

The first run downloads Hugging Face weights to the computer executing the job,
which may take meaningful disk/RAM on an 8 GiB Mac. Use a temporary worker
instead if necessary. The job stores individual `.npy` batches and batch ID
files, then `vectors.npy`, `paper_ids.json`, and `manifest.json` under a
content-addressed `embeddings/openalex-<query-hash>/specter2-<digest>/` prefix.

After an interruption, rerun the same command with the **same batch size**.
Completed batches will be reused. Different batch sizes change batch boundaries.
Do not change adapter/model versions midway through an experiment: retain the
manifest and pin tested dependency versions for reproducibility.

No OpenSearch index is modified. PR5 will read the manifest and index saved vectors.
