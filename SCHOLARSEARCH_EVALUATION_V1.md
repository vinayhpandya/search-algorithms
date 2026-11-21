# ScholarSearch: evaluation contract and demo (PR — independent 15K evaluation)

**Target code baseline:** GitHub `main` at `36adb8a`. Additive files only. Historical `eval/` is not reused.

**Purpose:** version the real 15K corpus, author and freeze **new** evaluation queries,
collect candidate rankings, create a blind judgment sheet, and score only when
required judgments exist. This module is AWS-job-friendly (all CLI commands
work from a container) but **does not provision AWS**, train a model, build an
index, produce relevance judgments, or claim real NDCG values.

## Quick verification: no AWS and no model dependencies

```bash
uv run python -m unittest discover -s tests -p 'test_scholarsearch_eval_v1.py' -v
uv run python -m scholarsearch.evaluation_v1 demo --out artifacts/evaluation/toy-demo
cat artifacts/evaluation/toy-demo/toy_report.json
```

The `toy_report.json` explicitly says `TOY_DATA_DO_NOT_PRESENT_AS_REAL`.
This demonstrates the full scoring and validation contract, not the 15K ranking quality.

## Step 1: validate the actual S3 corpus (read-only)

```bash
uv run --env-file .env python -m scholarsearch.evaluation_v1 snapshot \
  --root s3://scholarsearch-1a1f8b35f8/scholarsearch/experiments/experiment-b \
  --query 'malware detection' \
  --out artifacts/evaluation/v1/corpus_manifest.json
```

Uses **existing** `scholarsearch.index.load_papers`, `curated_uri`, and
`scholarsearch.embeddings.source_digest`, checks count = 15,000 and digest =
`379edc7bf2f15f55`, stores all document IDs and prominent title terms.
Does not generate embeddings or create an index. Reads the Parquet via S3,
which may incur minor S3 request/transfer charges. It does not deploy any compute.

## Step 2: write and review NEW queries

```bash
uv run python -m scholarsearch.evaluation_v1 template \
  --manifest artifacts/evaluation/v1/corpus_manifest.json \
  --count 100 \
  --out artifacts/evaluation/v1/queries_draft.tsv
```

Use corpus terms and examine title/abstract examples to draft 100 **genuinely
researcher-like questions** across topical, exploratory, and specific-paper
intents. Fill the `query` and `intent` fields. The template intentionally
starts BLANK, rather than passing fabricated synthetic queries off as genuine
researcher queries. A reviewer must set `reviewed=yes` on each row.

Once reviewed:

```bash
uv run python -m scholarsearch.evaluation_v1 freeze \
  --manifest artifacts/evaluation/v1/corpus_manifest.json \
  --input artifacts/evaluation/v1/queries_draft.tsv \
  --dev 40 --seed 42 \
  --out artifacts/evaluation/v1/queries.jsonl
```

Writes `queries.jsonl` (40 dev + 60 test) and `benchmark_manifest.json`
with query checksum, corpus digest and a statement that legacy eval is excluded.
Refuses blank, duplicate and unreviewed queries. Does not overwrite existing
frozen queries unless `--force` is explicit. The test set is held out from tuning.

## Step 3: retrieval jobs produce standard TREC run files

Each method (BM25, SPECTER2 dense, hybrid RRF, ColBERT) should output:

```
q001 Q0 W12345 1 12.1 bm25
q001 Q0 W67890 2 11.2 bm25
```

Run files must come from the **same 15K corpus and frozen query set**. Validation:

```bash
uv run python -m scholarsearch.evaluation_v1 validate-run \
  --manifest artifacts/evaluation/v1/corpus_manifest.json \
  --queries artifacts/evaluation/v1/queries.jsonl \
  --runs artifacts/evaluation/v1/runs/bm25.trec
```

Rejects results with unknown query IDs, unknown paper IDs, duplicated papers,
missing rank numbers, non-finite scores or mixed run tags. Retrieval adapters
are not included in this PR, so actual 15K method comparison isn't yet runnable.

## Step 4: produce blinded candidate pool for judgments

```bash
uv run python -m scholarsearch.evaluation_v1 pool \
  --manifest artifacts/evaluation/v1/corpus_manifest.json \
  --queries artifacts/evaluation/v1/queries.jsonl \
  --runs artifacts/evaluation/v1/runs/bm25.trec artifacts/evaluation/v1/runs/dense.trec \
  --depth 10 --allow-id-only \
  --out artifacts/evaluation/v1/pool_ids.tsv

uv run --env-file .env python -m scholarsearch.evaluation_v1 export-corpus \
  --manifest artifacts/evaluation/v1/corpus_manifest.json \
  --out artifacts/evaluation/v1/papers.jsonl

uv run python -m scholarsearch.evaluation_v1 annotate-pool \
  --manifest artifacts/evaluation/v1/corpus_manifest.json \
  --queries artifacts/evaluation/v1/queries.jsonl \
  --pool artifacts/evaluation/v1/pool_ids.tsv \
  --corpus-jsonl artifacts/evaluation/v1/papers.jsonl \
  --out artifacts/evaluation/v1/review.tsv
```

`review.tsv` has **no method names**; the separate
`pool_ids.tsv.provenance.json` retains where candidates originated.
Reviewers see query/title/abstract and explicitly set `relevance` (0,1,2,3),
`judge` ID and optional `notes`. Blank is unjudged, NEVER irrelevant.
Use multiple retrieval methods to reduce pool bias. Request independent
spot-checks and resolve disagreements on a documented subset before freezing qrels.

## Step 5: export qrels and score with coverage gate

```bash
uv run python -m scholarsearch.evaluation_v1 judgments \
  --manifest artifacts/evaluation/v1/corpus_manifest.json \
  --queries artifacts/evaluation/v1/queries.jsonl \
  --reviewed artifacts/evaluation/v1/review.tsv \
  --out artifacts/evaluation/v1/qrels.tsv

uv run python -m scholarsearch.evaluation_v1 score \
  --manifest artifacts/evaluation/v1/corpus_manifest.json \
  --queries artifacts/evaluation/v1/queries.jsonl \
  --runs artifacts/evaluation/v1/runs/bm25.trec artifacts/evaluation/v1/runs/dense.trec \
  --qrels artifacts/evaluation/v1/qrels.tsv \
  --split test --k 10 \
  --out artifacts/evaluation/v1/report.json
```

Outputs per-method `NDCG@10` (standard global judged ideal per query) and
`MRR@10` (first relevance>=2) **only if** all top-10 returned documents have
explicit judgments, the method returns at least 10 results, and there is at
least one known relevant document. Otherwise aggregate scores are `null`,
status is `INCOMPLETE_JUDGMENTS_NO_AGGREGATE_SCORE`, with missing coverage.
These remain *pooled* relevance scores, not estimates of exhaustive recall.

## AWS automation integration (future deployment, no infrastructure yet)

1. EventBridge triggers Step Functions.
2. Step Functions checks/ingests the *new* source snapshot (separately from benchmark).
3. AWS Batch generates missing embeddings and builds indexes on ephemeral workers.
4. Batch retrieval workers each emit TREC runs plus timing/cost artifacts to S3.
5. Batch `validate-run`, then `score` with immutable benchmark/qrels manifests.
6. Results are published only if coverage gates succeed; failures preserve logs and do not rewrite benchmark.

**Frozen relevance track:** Repeat the original corpus/query/qrels/index settings
for direct quality comparability. On a growing corpus, run cost/latency/ANN
benchmarks independently; update relevance judgment coverage *before* claiming
quality improvements. Periodically version a fresh relevance benchmark.

## Present to the group next week

- **Already measured:** 15K SPECTER2 ANN benchmarks; HNSW/IVF-PQ trade-offs.
- **Demonstrable today:** corpus snapshot validation and strict evaluation
  contract, toy end-to-end scoring demo, rejection tests.
- **Not measured yet:** 15K BM25/dense/ColBERT NDCG, 100 real reviewed queries,
  human relevance qrels, scheduled AWS workflow.
- **Next engineering milestone:** retrieval run adapter(s) + complete judged pool.

**Do not report toy NDCG as real evaluation performance.**
