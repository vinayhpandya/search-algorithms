# Stage 1 + Stage 2 additive prototype

This patch adds an independently runnable **OpenAlex ingestion → curated corpus → ANN benchmarking** path to `search-algorithms`. It does not replace the existing arXiv/Semantic Scholar pipeline, OpenSearch scripts, SPECTER2 embeddings, RQ-VAE, or current evaluations. No cloud resources are provisioned by the patch.

## Setup (run at repository root)

```bash
# Adds packages without changing your existing uv.lock until you choose to commit it
uv pip install --python .venv/bin/python requests numpy polars boto3 faiss-cpu
# if no .venv exists, first run: uv venv --python 3.11
python -m unittest discover -s tests -p 'test_stage12.py' -v
```

Note: On macOS, Faiss wheels / architecture support can vary. A CPU Linux VM or cloud notebook is recommended for larger ANN experiments. Running only Stage 1 does not require Faiss.

## Stage 1: ingestion (small local run)

```bash
export OPENALEX_API_KEY='your-key'   # if required by your API plan; never commit
python -m scholarsearch.ingest --root artifacts --query 'malware detection' --max-pages 2 --page-size 100
python -m scholarsearch.curate --root artifacts --query 'malware detection' --format jsonl
# Optional: compressed columnar dataset; pip install polars
python -m scholarsearch.curate --root artifacts --query 'malware detection' --format parquet
```

- `--max-pages` is the number of pages **this invocation**, not a lifetime limit; rerun to resume.
- Each raw API response is saved *before* the checkpoint advances.
- The checkpoint records the current OpenAlex cursor. A crash after page save but before checkpoint may re-fetch and overwrite one page safely.
- The job is intentionally serial; it handles HTTP 429/5xx with bounded retries and honors numeric Retry-After.
- If the source changes between repeated runs, use a new query or fresh output directory.
- The script fetches metadata/abstracts, not full PDFs. It does not evade provider quotas. Verify current OpenAlex terms, limits, and API-key requirements.

### Cloud storage instead of local disk

```bash
# Create a private bucket in your chosen AWS account first and set up AWS credentials
python -m scholarsearch.ingest --root s3://YOUR_BUCKET/scholarsearch --query 'malware detection' --max-pages 2
python -m scholarsearch.curate --root s3://YOUR_BUCKET/scholarsearch --query 'malware detection' --format parquet
```

This uses ordinary `boto3` credentials. For Azure or GCP, use a local working directory for now, or build storage adapters for Blob Storage / GCS; an S3 URL requires AWS or S3-compatible storage.

## Stage 2: benchmark Faiss Flat vs IVF-Flat vs HNSW

```bash
python examples/make_vectors.py --count 2000 --dimension 128
python -m scholarsearch.ann --vectors artifacts/example_vectors.npy --nlist 32 --nprobes 1 4 16 --out artifacts/benchmarks/toy.json
```

The toy dataset only exercises the benchmark. Replace it with your existing SPECTER2 embeddings stored as a `(N, D)` `float32` `.npy` array; preserve a paper-ID manifest in the same row order. Use the *same* vectors across exact, IVF, and HNSW. Larger experiments belong on a CPU VM because your M1 has 8 GB RAM.

`recall_at_k` here is ANN neighbor recall **relative to exact cosine search**, not relevance NDCG. Query vectors are held out from the indexed base. `mean_query_ms` measures Faiss search only; it excludes model inference and networking. Collect repeated runs and machine metadata before making claims about p95 or costs. Increase corpus size before drawing conclusions: 2,000 random vectors are just a smoke test.

### OpenSearch IVF

```bash
python -m scholarsearch.opensearch_ivf --dimension 768 --nlist 32
```

This writes a **mapping experiment template**, it does not create an OpenSearch index or train an IVF model. Your OpenSearch 2.15 deployment may require training through its Train Model API and a `model_id`-based field mapping. Keep your working HNSW index untouched. Benchmark Faiss IVF first; integrate trained OpenSearch IVF in a separate follow-up once version compatibility is checked.

## Cloud / budget guardrails

- **Don't deploy cloud infrastructure until the local smoke tests succeed.**
- Blob/GCS is fine if you choose Azure/GCP; don't assume the S3 adapter works there.
- Use a stopped/started CPU VM for OpenSearch; training experiments can use temporary VMs.
- Avoid public unauthenticated OpenSearch. The repository's existing local Compose file has security disabled; do not expose port 9200 to the internet.
- Set a cloud billing alert and exclude model downloads, data egress, and persistent disks from naive compute-only estimates.
- Add `artifacts/` to `.git/info/exclude` locally or `.gitignore` to avoid accidentally committing datasets, credentials, and vectors.

## Integration follow-up

To connect Stage 1 to your existing index, map the canonical columns (`id`, `title`, `abstract`, `citations`, etc.) to the schema your `index/index_papers.py` uses, and generate the same SPECTER2 vectors as the existing corpus. This patch deliberately avoids changing existing index mappings or evaluation judgments without verifying exact versions and datasets.
