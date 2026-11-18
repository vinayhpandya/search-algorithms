# PR6 Experiment B — 10k–15k-paper ANN scaling

This is deliberately **separate from the original 300-paper corpus**. No EC2 is
required to ingest or benchmark; SPECTER2 embedding can run locally or on a
temporary worker. S3 request/storage costs and OpenAlex usage limits apply.

## 0. Prerequisites

From the repo root, ensure `.env` has `AWS_PROFILE=scholarsearch-terraform`,
`SCHOLARSEARCH_S3_ROOT=s3://scholarsearch-1a1f8b35f8/scholarsearch`,
`SCHOLARSEARCH_QUERY="malware detection"`, `OPENALEX_API_KEY`, and `HF_TOKEN`.
Run `aws login --profile scholarsearch` when AWS login expires.

## 1. Inspect and ingest in bounded chunks

```bash
uv run --env-file .env python -m scholarsearch.experiment_b plan
uv run --env-file .env python -m scholarsearch.experiment_b ingest --pages 10 --page-size 100
uv run --env-file .env python -m scholarsearch.experiment_b status
```

Each `ingest` call requests at most 1,000 results and writes every page before
advancing its checkpoint. Repeat it, e.g. ~15 times for up to 15,000 raw
results. **15,000 raw results is not necessarily 15,000 unique papers**. Check
OpenAlex availability; if there are fewer matching papers, broaden the query
in a *new experiment name* rather than pretending you hit the target.

## 2. Freeze a reproducible, deduplicated corpus

```bash
uv run --env-file .env python -m scholarsearch.experiment_b freeze --min-papers 10000
```

This curates the current pages into Parquet, verifies the unique-paper count,
and writes `corpus-freeze.json`. After that this experiment's ingest command
refuses to append pages. **Do not rerun the older `scholarsearch.ingest` or
`scholarsearch.curate` directly with this experiment root after freezing**;
those old commands don't enforce the freeze guard. Consider S3 versioning or
an immutable bucket policy if hard immutability is required.

The output shows `experiment_root`; for the default root this is:

`s3://scholarsearch-1a1f8b35f8/scholarsearch/experiments/experiment-b`

## 3. Generate embeddings after freeze

Your local 8-GB Mac can attempt small batches but 10k–15k papers may take a
long time. To run locally with the dedicated Transformers-4 environment:

```bash
set -a; source .env; set +a
.venv-embed/bin/python -m scholarsearch.embeddings \
  --root s3://scholarsearch-1a1f8b35f8/scholarsearch/experiments/experiment-b \
  --query "malware detection" --batch-size 4 --device cpu
```

PR4 writes S3 checkpoints. **Keep batch size stable across restarts**, because
PR4 batch filenames use starting row numbers. To avoid writing thousands of
small S3 objects, a separate bulk-embedding optimization is recommended before
running the full 15k pipeline; do not increase batch size blindly on 8-GB RAM.

## 4. Benchmark after embeddings complete

```bash
uv run --env-file .env python -m scholarsearch.ann_benchmark \
  --root s3://scholarsearch-1a1f8b35f8/scholarsearch/experiments/experiment-b \
  --query "malware detection" --queries 200 --k 10 \
  --nlist 128 --nprobes 1 4 8 16 32 \
  --ef-search 16 32 64 128 256 --repeats 5 \
  --out artifacts/benchmarks/pr6-experiment-b.json
```

Check the benchmark's `skipped` field. IVF-PQ may be skipped if the training
set is below the minimum; ~10k *total* with 200 held-out leaves fewer than
9,984 base vectors. Target >10,184 total, preferably 15k unique, to cover it.
This benchmark measures ANN approximation recall, **not human relevance**.
