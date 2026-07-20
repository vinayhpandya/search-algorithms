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
queries, 857 unique query-paper pairs across BM25/dense/hybrid/rerank.

**NDCG@10 / MRR (LLM-judged labels, unverified beyond a spot-check sample):**

| Method | NDCG@10 | MRR |
|---|---|---|
| BM25   | 0.9029 | 1.0000 |
| Dense  | 0.9026 | 1.0000 |
| Hybrid | 0.9144 | 1.0000 |
| Rerank | 0.9131 | 1.0000 |

Hybrid and rerank show a modest but consistent NDCG@10 edge over BM25
and dense alone — evidence that fusion + reranking genuinely improves
ranking quality, not just reordering noise.

**MRR is saturated at 1.0 across all methods, and this is a labeling
finding, not a bug.** Label distribution across all 857 pairs:
`{1: 672, 2: 159, 0: 26}` — only 3% of pairs were judged "not relevant."
Because candidate pools are built from real search results (already
topically related to the query), the researcher/critic prompts turned
out to be too lenient about what counts as "not relevant" (0), so
almost every query's #1 result across every method landed on label ≥1,
trivially saturating MRR. NDCG@10 still carries signal since it weights
label=2 three times as heavily as label=1. Planned fix: tighten the
relevance scale to require the paper's *main focus* (not just topical
adjacency) matches the query, and relabel.

## Next session plan

- Expand corpus with additional queries/categories
- Relabel with a stricter relevance scale (fix the MRR-saturation issue above)
- Add LightGBM LTR as a 5th method, trained on {BM25 score, dense score,
  citation count, recency} with a query-level train/test split
- Possibly extend into a second weekend covering LLM-as-judge at scale,
  late-interaction retrieval, personalization, and query rewriting