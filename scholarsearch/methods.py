"""Retrieval methods for the experiment runner: first-stage search plus optional rerankers.

First stage (cheap, numpy/Faiss only):
  bm25         Python BM25 from eval_retrieve (title x3 + abstract)
  dense        exact SPECTER2 inner product (same as eval_retrieve)
  dense_hnsw   SPECTER2 through a Faiss HNSW index (approximate)
  hybrid       RRF(bm25, dense)
  hybrid_hnsw  RRF(bm25, dense_hnsw)

bm25 / dense / hybrid reproduce eval_retrieve.retrieve exactly (tested), so earlier
judgments and published numbers stay valid.

Rerankers re-order the top `pool` papers of a first-stage method and are named
"<base>+<reranker>", e.g. "hybrid+cross_encoder":
  cross_encoder  sentence-transformers CrossEncoder reads (query, paper) jointly
  colbert        ColBERT late interaction (fastembed): MaxSim over token embeddings

Latency: every method records per-query milliseconds. A reranked method's latency is
the base method's latency plus the reranker's time. ColBERT paper embeddings are
encoded once and cached *outside* the timed region, simulating a stored index (a later
step persists them); that one-off encoding cost is reported separately.
"""
from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np

FIRST_STAGE = ('bm25', 'dense', 'dense_hnsw', 'hybrid', 'hybrid_hnsw')
DEFAULT_MODELS = {'cross_encoder': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                  'colbert': 'colbert-ir/colbertv2.0'}


def _ms(t0: int) -> float:
    return (time.perf_counter_ns() - t0) / 1e6


def paper_text(hit: dict) -> str:
    title, abstract = (hit.get('title') or '').strip(), (hit.get('abstract') or '').strip()
    return f'{title}. {abstract}' if abstract else title


# --------------------------------------------------------------------------- rerankers

class CrossEncoderReranker:
    def __init__(self, model: str, batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.model, self.batch_size = CrossEncoder(model, device='cpu'), batch_size

    def prepare(self, hits: list[dict]) -> None:  # nothing to precompute: scores are joint
        pass

    def score(self, query: str, hits: list[dict]) -> list[float]:
        pairs = [(query, paper_text(h)) for h in hits]
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch_size,
                                                     show_progress_bar=False)]


class ColbertReranker:
    def __init__(self, model: str, batch_size: int = 32):
        from fastembed import LateInteractionTextEmbedding
        self.model, self.batch_size = LateInteractionTextEmbedding(model), batch_size
        self.cache: dict[str, np.ndarray] = {}
        self.encode_seconds = 0.

    def prepare(self, hits: list[dict]) -> None:
        """Encode papers not seen before. Called outside the timed region."""
        new = [h for h in hits if h['id'] not in self.cache]
        if not new:
            return
        t0 = time.perf_counter()
        for hit, emb in zip(new, self.model.embed([paper_text(h) for h in new],
                                                  batch_size=self.batch_size)):
            self.cache[hit['id']] = np.asarray(emb, dtype=np.float32)
        self.encode_seconds += time.perf_counter() - t0

    def score(self, query: str, hits: list[dict]) -> list[float]:
        q = np.asarray(next(iter(self.model.query_embed([query]))), dtype=np.float32)
        return [float((q @ self.cache[h['id']].T).max(axis=1).sum()) for h in hits]


RERANKERS = {'cross_encoder': CrossEncoderReranker, 'colbert': ColbertReranker}


# --------------------------------------------------------------------------- runner

def parse_methods(methods: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Split ['bm25', 'hybrid+colbert'] into first-stage names and (base, reranker) pairs."""
    first, reranked = [], []
    for name in methods:
        base, _, reranker = name.partition('+')
        if base not in FIRST_STAGE:
            raise ValueError(f'Unknown first-stage method {base!r}; choose from {FIRST_STAGE}')
        if reranker and reranker not in RERANKERS:
            raise ValueError(f'Unknown reranker {reranker!r}; choose from {sorted(RERANKERS)}')
        if reranker:
            reranked.append((base, reranker))
        first.append(base)
    if len(set(methods)) != len(methods):
        raise ValueError('Duplicate method names')
    return list(dict.fromkeys(first)), reranked


def build_hnsw(vectors: np.ndarray, *, m: int, ef_construction: int, ef_search: int):
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter_ns()
    index.add(vectors)
    build_ms = _ms(t0)
    index.hnsw.efSearch = ef_search
    return index, build_ms


def run_methods(papers: list[dict], doc_vectors: np.ndarray, queries: list[dict],
                query_vectors: np.ndarray, *, methods: list[str], k: int = 10,
                candidate_k: int = 50, pool: int = 100, hnsw: dict | None = None,
                models: dict | None = None, reranker_factories: dict | None = None) -> dict:
    """Return an eval_retrieve-compatible report ('UNJUDGED_RETRIEVAL_ONLY') for any methods."""
    from .eval_retrieve import BM25, _docs
    from .embeddings import source_digest
    from .search import rrf

    first, reranked = parse_methods(methods)
    needs = set(first)
    if needs & {'hybrid', 'hybrid_hnsw'}:
        needs.add('bm25')
    if 'hybrid' in needs:
        needs.add('dense')
    if 'hybrid_hnsw' in needs:
        needs.add('dense_hnsw')
    if k < 1 or candidate_k < k or pool < k:
        raise ValueError('Require candidate_k >= k and pool >= k >= 1')
    ids = [str(p.get('id') or '') for p in papers]
    if not all(ids) or len(ids) != len(set(ids)) or len(ids) != len(doc_vectors):
        raise ValueError('Missing/duplicate IDs or inconsistent corpus/vector count')
    if query_vectors.shape != (len(queries), doc_vectors.shape[1]):
        raise ValueError('Query vectors do not match queries / document dimension')

    details, setup = {}, {}
    bm25 = BM25(papers) if 'bm25' in needs else None
    index = None
    hnsw = {'M': 16, 'ef_construction': 200, 'ef_search': 64, **(hnsw or {})}
    if 'dense_hnsw' in needs:
        index, setup['hnsw_build_ms'] = build_hnsw(doc_vectors, m=hnsw['M'],
                                                   ef_construction=hnsw['ef_construction'],
                                                   ef_search=hnsw['ef_search'])
        details['dense_hnsw'] = f"Faiss IndexHNSWFlat inner product, M={hnsw['M']}, efConstruction={hnsw['ef_construction']}, efSearch={hnsw['ef_search']}, 1 thread"
    factories = reranker_factories or RERANKERS
    models = {**DEFAULT_MODELS, **(models or {})}
    rerankers = {}
    for _, name in reranked:
        if name not in rerankers:
            t0 = time.perf_counter()
            rerankers[name] = factories[name](models[name])
            setup[f'{name}_load_seconds'] = round(time.perf_counter() - t0, 3)
            details[name] = f'{name} reranker, model {models[name]}, top {pool} of the base method'

    output = []
    for row, qvec in zip(queries, query_vectors):
        lists, lat = {}, {}
        if 'bm25' in needs:
            t0 = time.perf_counter_ns()
            lists['bm25'] = _docs(bm25.search(row['query'], candidate_k), papers)
            lat['bm25'] = _ms(t0)
        if 'dense' in needs:
            t0 = time.perf_counter_ns()
            scores = doc_vectors @ qvec
            top = sorted(range(len(papers)), key=lambda i: (-float(scores[i]), ids[i]))[:candidate_k]
            lists['dense'] = _docs([(i, float(scores[i])) for i in top], papers)
            lat['dense'] = _ms(t0)
        if 'dense_hnsw' in needs:
            t0 = time.perf_counter_ns()
            sims, idx = index.search(qvec[None, :], candidate_k)
            lists['dense_hnsw'] = _docs([(int(i), float(s)) for i, s in zip(idx[0], sims[0]) if i >= 0], papers)
            lat['dense_hnsw'] = _ms(t0)
        for name, dense_name in (('hybrid', 'dense'), ('hybrid_hnsw', 'dense_hnsw')):
            if name in needs:
                t0 = time.perf_counter_ns()
                # limit=pool keeps the top-k identical to limit=k (sorted, then sliced)
                lists[name] = rrf([lists['bm25'], lists[dense_name]], k=60, limit=pool)
                lat[name] = lat['bm25'] + lat[dense_name] + _ms(t0)
        results = {name: lists[name][:k] for name in first}
        for base, name in reranked:
            candidates = lists[base][:pool]
            reranker = rerankers[name]
            reranker.prepare(candidates)
            t0 = time.perf_counter_ns()
            scores = reranker.score(row['query'], candidates)
            rerank_ms = _ms(t0)
            order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]['id']))
            method = f'{base}+{name}'
            results[method] = [{**candidates[i], 'score': scores[i], 'base_rank': i + 1}
                               for i in order[:k]]
            lat[method] = lat[base] + rerank_ms
        if 'hybrid' in lat:
            lat['hybrid_total'] = lat['hybrid']  # name used by earlier reports
        output.append({'query_id': row['query_id'], 'query': row['query'],
                       'methods': {m: results[m] for m in methods}, 'latency_ms': lat})
        print(f"{row['query_id']} done", flush=True)

    for name, reranker in rerankers.items():
        if hasattr(reranker, 'encode_seconds'):
            setup[f'{name}_paper_encoding_seconds'] = round(reranker.encode_seconds, 3)
            setup[f'{name}_papers_encoded'] = len(reranker.cache)
    details.update({
        'bm25': 'Python BM25, title tokens repeated 3x plus abstract',
        'dense': 'Exact normalized SPECTER2 inner product, not ANN',
        'hybrid': f'RRF k=60 on top candidate_k={candidate_k} BM25 and dense results',
        'hybrid_hnsw': f'RRF k=60 on top candidate_k={candidate_k} BM25 and HNSW dense results',
        'latency': 'Per-query search time; excludes query SPECTER2 encoding, S3 loading and index build. '
                   'Reranked = base + reranker time. ColBERT paper embeddings are precomputed (cached) '
                   'outside the timed region, as a stored index would be.'})
    return {'status': 'UNJUDGED_RETRIEVAL_ONLY', 'methods': methods,
            'method_details': {k_: v for k_, v in details.items()
                               if k_ in needs or k_ in rerankers or k_ == 'latency'},
            'setup': setup,
            'corpus': {'paper_count': len(papers), 'source_digest': source_digest(papers)},
            'query_count': len(queries), 'k': k, 'candidate_k': candidate_k, 'pool': pool,
            'created_at': datetime.now(timezone.utc).isoformat(), 'results': output}
