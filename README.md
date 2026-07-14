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