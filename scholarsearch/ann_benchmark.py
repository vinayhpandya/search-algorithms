"""PR6: reproducible CPU Faiss ANN benchmark over PR4's SPECTER2 artifacts.

Queries are held-out *paper vectors*, NOT separately encoded natural-language queries.
This measures index approximation recall, not search relevance (NDCG/MRR).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_env, setting
from .embeddings import artifact_prefix, MODEL, PAPER_ADAPTER
from .storage import get_bytes, join


def read_corpus(root: str, query: str) -> tuple[np.ndarray, dict]:
    """Load and validate the *completed* PR4 artifact from persistent S3."""
    # Avoid deriving the source digest from a possibly changed curated dataset.
    from .index import curated_uri, load_papers
    from .embeddings import source_digest

    papers = load_papers(curated_uri(root, query))
    if not papers:
        raise ValueError('Curated corpus is empty')
    prefix = artifact_prefix(root, query, source_digest(papers))
    manifest = json.loads(get_bytes(join(prefix, 'manifest.json')))
    ids = json.loads(get_bytes(manifest['paper_ids_uri']))
    vectors = np.load(io.BytesIO(get_bytes(manifest['vectors_uri'])), allow_pickle=False)
    if (manifest.get('source_digest') != source_digest(papers)
            or manifest.get('model') != MODEL or manifest.get('paper_adapter') != PAPER_ADAPTER
            or manifest.get('normalized') is not True):
        raise ValueError('PR4 artifact metadata mismatch')
    expected_ids = [str(p['id']) for p in papers]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError('Embedding paper IDs do not align with the curated corpus')
    validate_vectors(vectors, expected_count=len(ids), dimension=manifest['dimension'])
    return np.ascontiguousarray(vectors, dtype=np.float32), manifest


def validate_vectors(vectors: np.ndarray, *, expected_count: int | None = None,
                     dimension: int | None = None) -> None:
    if vectors.ndim != 2 or vectors.shape[0] < 3 or vectors.shape[1] == 0:
        raise ValueError('Expected an (N, D) array with at least 3 vectors')
    if expected_count is not None and vectors.shape[0] != expected_count:
        raise ValueError('Vector count differs from paper ID count')
    if dimension is not None and vectors.shape[1] != dimension:
        raise ValueError('Vector dimension differs from PR4 manifest')
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError('Expected finite float32 vectors')
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError('Expected L2-normalized embeddings')


def split_vectors(vectors: np.ndarray, *, seed: int, queries: int) -> tuple[np.ndarray, np.ndarray]:
    if queries <= 0 or queries >= len(vectors):
        raise ValueError('queries must be in [1, number_of_vectors - 1]')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vectors))
    return (np.ascontiguousarray(vectors[perm[queries:]], dtype=np.float32),
            np.ascontiguousarray(vectors[perm[:queries]], dtype=np.float32))


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    if found.shape != truth.shape or found.ndim != 2:
        raise ValueError('Expected matching (query_count, k) result arrays')
    # Missing (-1) ids never count as correct results.
    hits = sum(len(set(int(z) for z in row if z >= 0) &
                   set(int(z) for z in ref if z >= 0))
               for row, ref in zip(found, truth))
    return hits / truth.size


def percentile_ms(samples: list[float]) -> dict:
    return {'p50_ms': round(float(np.percentile(samples, 50)), 5),
            'p95_ms': round(float(np.percentile(samples, 95)), 5),
            'mean_ms': round(float(np.mean(samples)), 5)}


def collect(index, q: np.ndarray, k: int, truth: np.ndarray, *, repeats: int,
            warmups: int, faiss) -> dict:
    # Each measurement is one Faiss search on one vector (not batch throughput).
    for _ in range(warmups):
        for j in range(len(q)):
            index.search(q[j:j+1], k)
    samples = []
    for _ in range(repeats):
        for j in range(len(q)):
            t0 = time.perf_counter_ns()
            index.search(q[j:j+1], k)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
    _, found = index.search(q, k)
    return {'recall_at_k': round(recall_at_k(found, truth), 6),
            **percentile_ms(samples),
            'index_bytes': len(faiss.serialize_index(index))}


def run_benchmark(vectors: np.ndarray, *, query_count: int = 50, k: int = 10,
                  seed: int = 42, nlist: int = 16, nprobes: tuple[int, ...] = (1, 4, 16),
                  hnsw_m: int = 16, ef_search: tuple[int, ...] = (16, 64),
                  pq_m: int = 16, pq_bits: int = 8, repeats: int = 3,
                  warmups: int = 1, threads: int = 1) -> dict:
    import faiss

    validate_vectors(vectors)
    if min(k, query_count, nlist, hnsw_m, pq_m, pq_bits, repeats, threads) <= 0 or warmups < 0:
        raise ValueError('All benchmark parameters except warmups must be positive')
    base, q = split_vectors(vectors, seed=seed, queries=query_count)
    if len(base) < max(k, nlist):
        raise ValueError('Need at least max(k,nlist) base vectors')
    if vectors.shape[1] % pq_m:
        raise ValueError('Vector dimension must be divisible by PQ subquantizer count')
    if pq_bits > 8:
        raise ValueError('This PR6 implementation supports at most 8 bits per PQ code')
    if not nprobes or not ef_search or any(v <= 0 for v in (*nprobes, *ef_search)):
        raise ValueError('nprobes and ef_search must contain positive values')
    faiss.omp_set_num_threads(threads)
    dim = base.shape[1]
    rows = []
    skipped = []

    def record(method: str, index, *, build_ms: float, train_ms: float = 0,
               parameters: dict | None = None):
        rows.append({'method': method, 'parameters': parameters or {},
                     'training_ms': round(train_ms, 3), 'build_ms': round(build_ms, 3),
                     **collect(index, q, k, truth, repeats=repeats, warmups=warmups, faiss=faiss)})

    exact = faiss.IndexFlatIP(dim)
    t0 = time.perf_counter_ns()
    exact.add(base)
    exact_build = (time.perf_counter_ns() - t0) / 1e6
    _, truth = exact.search(q, k)
    record('exact-flat', exact, build_ms=exact_build)

    # IVF training requires considerably more samples than nlist; skip rather
    # than publishing misleading results from undertrained centroids.
    min_ivf_train = 39 * nlist
    if len(base) >= min_ivf_train:
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter_ns()
        ivf.train(base)
        ivf_train_ms = (time.perf_counter_ns() - t0) / 1e6
        t0 = time.perf_counter_ns()
        ivf.add(base)
        build_ms = (time.perf_counter_ns() - t0) / 1e6
        for probe in sorted(set(min(nlist, p) for p in nprobes)):
            ivf.nprobe = probe
            record('ivf-flat', ivf, train_ms=ivf_train_ms, build_ms=build_ms,
                   parameters={'nlist': nlist, 'nprobe': probe})
    else:
        skipped.append({'method': 'ivf-flat', 'reason': f'Requires >= {min_ivf_train} base vectors for {nlist} clusters'})

    # For 8-bit PQ, 256 codewords per subquantizer: allow >=39*256
    # training examples, and enough examples to train the coarse IVF clusters.
    min_pq_train = max(min_ivf_train, 39 * (1 << pq_bits))
    if len(base) >= min_pq_train:
        pq = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, pq_m, pq_bits, faiss.METRIC_INNER_PRODUCT)
        t0 = time.perf_counter_ns()
        pq.train(base)
        pq_train_ms = (time.perf_counter_ns() - t0) / 1e6
        t0 = time.perf_counter_ns()
        pq.add(base)
        pq_build_ms = (time.perf_counter_ns() - t0) / 1e6
        for probe in sorted(set(min(nlist, p) for p in nprobes)):
            pq.nprobe = probe
            record('ivf-pq', pq, train_ms=pq_train_ms, build_ms=pq_build_ms,
                   parameters={'nlist': nlist, 'nprobe': probe, 'pq_m': pq_m, 'pq_bits': pq_bits})
    else:
        skipped.append({'method': 'ivf-pq', 'reason': f'Requires >= {min_pq_train} base vectors for PQ training'})

    hnsw = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter_ns()
    hnsw.add(base)
    hnsw_build_ms = (time.perf_counter_ns() - t0) / 1e6
    for ef in sorted(set(ef_search)):
        hnsw.hnsw.efSearch = ef
        record('hnsw', hnsw, build_ms=hnsw_build_ms,
               parameters={'M': hnsw_m, 'efSearch': ef})
    return {'created_at': datetime.now(timezone.utc).isoformat(),
            'dataset': {'total': len(vectors), 'base': len(base), 'held_out_paper_queries': len(q),
                        'dimension': dim, 'seed': seed},
            'protocol': {'metric': 'cosine / normalized inner product', 'k': k,
                         'repeats': repeats, 'warmups': warmups, 'faiss_threads': threads,
                         'ground_truth': 'exact IndexFlatIP on base',
                         'latency_scope': 'single-query Faiss search only, not model or network'},
            'results': rows, 'skipped': skipped,
            'limitations': 'Held-out paper vectors are NOT user queries; no relevance judgments. Small datasets are smoke tests, not performance conclusions.'}


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=setting('SCHOLARSEARCH_S3_ROOT'))
    parser.add_argument('--query', default=setting('SCHOLARSEARCH_QUERY', 'malware detection'))
    parser.add_argument('--queries', type=int, default=50)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--nlist', type=int, default=16)
    parser.add_argument('--nprobes', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--ef-search', type=int, nargs='+', default=[16, 64])
    parser.add_argument('--pq-m', type=int, default=16)
    parser.add_argument('--pq-bits', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', default='artifacts/benchmarks/pr6-ann.json')
    args = parser.parse_args()
    vectors, manifest = read_corpus(args.root, args.query)
    result = run_benchmark(vectors, query_count=args.queries, k=args.k,
                           nlist=args.nlist, nprobes=tuple(args.nprobes),
                           ef_search=tuple(args.ef_search), pq_m=args.pq_m,
                           pq_bits=args.pq_bits, repeats=args.repeats, threads=args.threads)
    result['source'] = {'uri': manifest['vectors_uri'], 'digest': manifest['source_digest'],
                        'model': manifest['model'], 'paper_adapter': manifest['paper_adapter']}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
