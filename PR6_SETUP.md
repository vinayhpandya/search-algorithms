# PR6 — Faiss ANN experiments (exact, IVF-Flat, IVF-PQ, HNSW)

This PR adds `scholarsearch/ann_benchmark.py` and `tests/test_pr6.py`. It **does not change** your PR5 search service, query encoder, S3 artifacts, or Terraform. PR1–2 already had `scholarsearch/ann.py`; PR6 provides a separate benchmark entrypoint with more rigorous metrics and manifest validation.

## Phase A — run a correctness/smoke benchmark on existing 300 embeddings

From repository root:

```bash
uv add faiss-cpu
uv run --env-file .env python -m unittest discover -s tests -p 'test_pr6.py' -v
uv run --env-file .env python -m scholarsearch.ann_benchmark \
  --queries 50 --k 10 --nlist 4 --nprobes 1 2 4 \
  --ef-search 16 64 --repeats 3 \
  --out artifacts/benchmarks/pr6-300.json
```

Faiss is local CPU code; **no EC2/OpenSearch/SSM tunnel is needed** for this phase. The benchmark downloads `vectors.npy` and its manifest from the existing persistent S3 bucket using `.env` AWS settings. If AWS CLI reports expired credentials, `aws login --profile scholarsearch` first.

Outputs include Recall@10 against exact IndexFlatIP, single-query P50/P95/mean search latency (ms), Faiss serialized index sizes (bytes), add/build times (ms), training times (ms), experiment settings, and skipped methods.

**Expected on 300 papers:** IVF-Flat and IVF-PQ may be skipped: Faiss recommends training data proportional to the number of centroids (`~39*nlist`) and PQ codewords (`~39*2^bits`). With 50 held-out queries, only 250 remain for indexing. The `--nlist 4` option needs at least 156 base vectors for IVF-Flat; IVF-PQ with 8 bits needs at least 9,984. Results on 300 vectors are a *code-validation experiment*, not a credible ANN performance comparison.

## Phase B — scale up before comparing algorithms

1. Ingest and curate ~15k–30k **distinct real papers** using your existing bounded OpenAlex ingestion pipeline (adjust the ingestion job's `--max-pages`/`--page-size`; existing checkpointing prevents restarting). Ensure curated dataset and digest are regenerated; do not mix embeddings from different corpus snapshots.
2. Run PR4's existing resumable SPECTER2 embedding worker for the new dataset. This can be costly and slow on an 8 GB M1: estimate volume and choose temporary compute only if needed. All results live in S3; no OpenSearch is required.
3. Re-run PR6 on the larger corpus. For example:

```bash
uv run --env-file .env python -m scholarsearch.ann_benchmark \
  --queries 100 --k 10 --nlist 64 --nprobes 1 4 16 64 \
  --ef-search 16 32 64 128 --repeats 5 \
  --out artifacts/benchmarks/pr6-large.json
```

You can change `--pq-m` and `--pq-bits` if desired (dimension 768 must be divisible by `pq_m`). At 15k–30k paper vectors IVF-PQ 8-bit has enough training examples. Results depend on hardware and are not comparable to OpenSearch timings, which include different engines and server/network overhead.

## Experimental protocol / caveats

- Hold out `--queries` distinct *paper vectors* as queries; build all indexes on the remaining vectors. Fixed RNG seed 42. Ground truth: exact inner-product search on the same base. Since embeddings are unit-normalized, inner product ranks cosine similarity.
- These held-out papers are a proxy workload: **this is ANN recall, not actual relevance quality**. To compare BM25/dense/hybrid quality later, create a labeled query set and measure nDCG@10, Recall@10, and MRR.
- Faiss CPU thread count is forced to 1 for repeatable comparisons. Runs measure search calls only; **no encoder, S3, Python app or OpenSearch network costs**. `index_bytes` is serialized index size, not actual process RSS. Build time is measured separately from training time.
- On tiny corpora, timings are noisy; use several repetitions and check P50/P95. Avoid claiming HNSW or IVF performance superiority until dataset size and workload are representative.
- Files in `artifacts/benchmarks/` can be committed as benchmark evidence after reviewing contents. No credentials are stored.

When using the temporary AWS OpenSearch for other experiments, clean up via `infra/aws/ephemeral/terraform destroy` after confirming the state and resources. PR6 itself incurs only S3 read requests.
