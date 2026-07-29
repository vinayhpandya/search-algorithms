# search-algorithms — Malware Classification Paper Search

Builds a search engine over arXiv malware-classification papers,
progressing through the course's core techniques and comparing them
head-to-head: **BM25 → dense (SPECTER2) → hybrid (RRF) → cross-encoder
rerank → LightGBM LTR**, evaluated with NDCG@10 / MRR at each stage.

## Why this structure

- **OpenSearch, not a bare vector DB** — supports BM25 and k-NN vector
  search in one engine, so lexical and semantic search share the same
  index and can be fused natively (hybrid RRF queries).
- **SPECTER2 embeddings** — trained specifically on scientific paper
  title+abstract pairs via citation graphs, should outperform a generic
  sentence embedding model on this domain.
- **Citation count as an LTR feature** — pulled from Semantic Scholar
  since arXiv itself doesn't expose it.

## Project layout

```
data/           collection + cleaning scripts, raw metadata (small, committed)
  raw/          papers.json — collected arXiv metadata + citation counts
  embeddings/   precomputed SPECTER2 vectors (backed up alongside the index)
index/          OpenSearch mapping + indexing scripts
  create_index.py   defines the BM25 + knn_vector schema
  index_papers.py   bulk-loads papers.json into OpenSearch
search/         bm25.py, dense.py, hybrid.py, rerank.py, ltr.py
  bm25.py       lexical baseline (multi_match, title^2 + abstract)
eval/           labeled queries + NDCG/MRR scoring
rag/            optional QA layer (stretch goal)
notebooks/      exploration, result plots
docker-compose.yml   OpenSearch + Dashboards, single-node local setup
```

## Status

**Weekend 1 — core retrieval pipeline**
- [x] Repo scaffold, Docker/OpenSearch setup
- [x] Data collection (arXiv `arxiv` client library + Semantic Scholar citations)
- [x] OpenSearch indexing (BM25 fields + reserved k-NN vector field)
- [x] BM25 baseline search
- [x] SPECTER2 embeddings + dense (k-NN) search
- [x] Hybrid search (Reciprocal Rank Fusion)
- [x] Cross-encoder reranking
- [x] Evaluation harness: 40 queries, 857 candidate query-paper pairs
- [x] LLM-judged relevance labels (researcher + critic), human spot-check sample
- [x] NDCG@10 / MRR comparison across BM25 / dense / hybrid / rerank
- [ ] LightGBM LTR (planned for next session, alongside expanding the corpus)

**Not yet started**
- [ ] Embedding-based clustering / semantic ID exploration
- [ ] RAG QA layer
- [ ] Weekend 2 additions: LLM-as-judge at larger scale, late-interaction
      (ColBERT-style) retrieval, personalization/signals boosting, query
      rewriting

## Findings so far

Current corpus: ~194 → grown to a larger set via broadened category
filter (`cs.CR`/`cs.LG`/`cs.CV`) and 16 search queries. Eval set: 40
queries, 857 unique query-paper pairs across BM25/dense/hybrid/rerank,
labeled via researcher+critic LLM judging (critic's rating is ground
truth) with a stratified human spot-check sample.

### Full eval set (40 queries) — BM25 / dense / hybrid / rerank

| Method | NDCG@10 | MRR(≥1) | MRR(≥2) |
|---|---|---|---|
| BM25   | 0.9029 | 1.0000 | 0.6094 |
| Dense  | 0.9026 | 1.0000 | 0.5763 |
| Hybrid | 0.9144 | 1.0000 | 0.7261 |
| Rerank | 0.9131 | 1.0000 | 0.7504 |

**MRR(≥1) is saturated at 1.0 for every method — this is a metric
choice issue, not a bug.** Only 26/857 pairs (3%) were labeled "not
relevant" (0), so nearly every query's rank-1 result across every
method has label ≥1, trivially maxing out MRR(≥1). Switching to
MRR(≥2) ("how often is the #1 result *highly* relevant, not just
somewhat") restored real discrimination between methods without
needing to relabel anything — confirms the labels had sufficient
signal all along, MRR(≥1) was just the wrong lens for this label
distribution.

**Dense alone underperforms BM25 on MRR(≥2)** (0.5763 vs 0.6094)
despite near-identical NDCG@10 — suggests SPECTER2 embeddings surface
broadly topical neighbors well (decent NDCG) but are less precise than
exact keyword matching at pinpointing the single best match at rank 1.
Hybrid and rerank both correct for this, each outperforming their
individual inputs — evidence hybrid/rerank are doing real work, not
just reshuffling noise.

### LTR — held-out test split (12 queries LTR never trained on)

Trained LightGBM `LGBMRanker` (LambdaMART) on 5 features
(bm25_score, dense_score, citation_count, year, title_exact_match)
using the same critic-labeled ground truth, with a query-level
train/test split (28 train / 12 test queries) to avoid leakage.

| Method | NDCG@10 | MRR(≥1) | MRR(≥2) |
|---|---|---|---|
| BM25   | 0.9085 | 1.0000 | 0.5653 |
| Dense  | 0.9001 | 1.0000 | 0.5378 |
| Hybrid | 0.9123 | 1.0000 | 0.7028 |
| Rerank | 0.9307 | 1.0000 | 0.8125 |
| **LTR**| 0.8878 | 1.0000 | 0.5833 |

**LTR underperformed hybrid and rerank**, landing roughly on par with
plain BM25/dense. Root cause, not a bug: LTR is the only method here
learning a relevance function *from scratch*, from just **28 training
queries** and **5 scalar features** — it never reads the actual title
or abstract text. By contrast, hybrid needs no training data at all
(RRF is a fixed formula over two already-strong rankings), and rerank
uses a cross-encoder *pretrained on hundreds of thousands of real
query-document pairs* (MS MARCO) that reads full text jointly with the
query — it never needed your training queries in the first place. This
isn't a fair fight by construction: LTR earns its keep in production
systems with large-scale real engagement data (clicks, purchases),
which this project doesn't have. A small-data LTR model losing to a
pretrained cross-encoder is a realistic, expected outcome, not a
failure to fix.

### Quality vs. latency

Latency measured across all 40 eval queries, 5 runs each (first
discarded as warmup), on a local machine (no GPU). LTR isn't included
here -- it operates on precomputed features rather than a live search
path, so "latency" would mean something different (a fast lookup +
`model.predict()` call, not comparable to end-to-end search).

| Method | NDCG@10 | MRR(≥2) | Mean latency | P95 latency |
|---|---|---|---|---|
| BM25             | 0.9029 | 0.6094 | 3.3ms   | 4.5ms   |
| Dense             | 0.9026 | 0.5763 | 39.4ms  | 42.8ms  |
| Hybrid            | 0.9144 | 0.7261 | 56.6ms  | 60.7ms  |
| Late-interaction  | 0.9003 | 0.6951 | 12.4ms  | 13.6ms  |
| Rerank            | 0.9131 | 0.7504 | 183.6ms | 192.2ms |

**The clearest quality-vs-latency story: rerank is the best quality by
both metrics, but costs ~15x hybrid's latency and ~55x BM25's** —
consistent with cross-encoders being the most expressive (full
cross-attention between query and document) but most expensive method
here, since it can only run on a pre-filtered shortlist (hybrid's
top-50), never the whole corpus.

**Late-interaction is a genuinely interesting middle ground**: it
clearly beats dense on MRR(≥2) (0.6951 vs 0.5763 -- finds the *most*
relevant paper at rank 1 far more reliably) while being noticeably
*faster* than dense (12.4ms vs 39.4ms), not slower as might be
expected from doing per-token rather than per-document matching. This
is likely attributable to fastembed's ONNX Runtime backend being
faster than the plain PyTorch/`transformers` path `dense.py` uses for
SPECTER2, rather than late-interaction being inherently cheaper than
single-vector search -- worth noting as a possible confound rather
than a clean apples-to-apples engine comparison (different serving
stacks, not just different retrieval algorithms).

**Practical takeaway**: for a system needing sub-20ms latency, BM25 or
late-interaction are the strongest options here relative to their
quality. Hybrid is a reasonable middle ground if ~60ms is acceptable.
Rerank should be reserved for cases where its quality edge over hybrid
(NDCG +0.001, MRR(≥2) +0.024 -- modest here) is worth ~130ms extra --
in this project's results, that tradeoff is arguably not favorable,
though production systems with better-calibrated relevance labels or
larger quality gaps might see a clearer case for it.

**Possible next step**: feed the cross-encoder's own score into LTR as
an additional feature ("stacking" a strong pretrained signal into the
learned model) — a common real technique, untried here due to time.

## Status

**Weekend 1 — core retrieval pipeline**
- [x] Repo scaffold, Docker/OpenSearch setup
- [x] Data collection (arXiv `arxiv` client library + Semantic Scholar citations)
- [x] OpenSearch indexing (BM25 fields + reserved k-NN vector field)
- [x] BM25 baseline search
- [x] SPECTER2 embeddings + dense (k-NN) search
- [x] Hybrid search (Reciprocal Rank Fusion)
- [x] Cross-encoder reranking
- [x] Evaluation harness: 40 queries, 857 candidate query-paper pairs
- [x] LLM-judged relevance labels (researcher + critic), human spot-check sample
- [x] NDCG@10 / MRR comparison across BM25 / dense / hybrid / rerank
- [x] LightGBM LTR (query-level train/test split, 5 features) — underperformed
      hybrid/rerank; documented as an expected small-data/feature-richness
      finding, not a bug (see Findings above)

**Not yet started**
- [ ] Embedding-based clustering / semantic ID exploration
- [ ] RAG QA layer
- [x] Weekend 2 additions: LLM-as-judge at larger scale, 
- [x] late-interaction (ColBERT-style) retrieval -> ~~We will be using
      RAG-atoullie since it offers retrieval models out of the box~~
      **Update:** switched to Qdrant + fastembed after RAGatouille (and
      subsequently PyLate) both hit unresolvable dependency conflicts with
      the project's transformers/sentence-transformers versions.

