"""Reproducible IVF vs exact vs HNSW Faiss benchmark on precomputed embeddings."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np


def load_vectors(path, *, max_vectors=None):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError("Vectors must be a 2D .npy array of shape (N, dimension)")
    if max_vectors:
        arr = arr[:max_vectors]
    result = np.ascontiguousarray(arr, dtype="float32")
    if not np.isfinite(result).all():
        raise ValueError("Vectors must be finite")
    return result


def evaluate(x, *, k=10, nlist=32, nprobes=(1, 4, 16), queries=100, train_size=10000, seed=42, hnsw_m=16):
    import faiss
    if x.shape[0] < max(k + 2, nlist):
        raise ValueError("Need at least max(k+2, nlist) vectors")
    rng = np.random.default_rng(seed)
    # Use held-out documents as queries, not vectors present in the index.
    qcount = min(queries, max(1, x.shape[0] // 5))
    perm = rng.permutation(x.shape[0])
    q = x[perm[:qcount]].copy()
    base = x[perm[qcount:]].copy()
    if len(base) < nlist:
        raise ValueError("Not enough base vectors for the requested nlist")
    faiss.normalize_L2(base)
    faiss.normalize_L2(q)
    dim = base.shape[1]
    gt = faiss.IndexFlatIP(dim)
    gt.add(base)
    truth = gt.search(q, k)[1]
    samples = base[rng.choice(len(base), size=min(len(base), train_size), replace=False)]

    def bench(name, index, probes=None):
        if probes is not None:
            index.nprobe = probes
        start = time.perf_counter()
        found = index.search(q, k)[1]
        elapsed = (time.perf_counter() - start) * 1000
        matched = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(found, truth))
        return {"method": name, "nprobe": probes, "recall_at_k": round(matched / (qcount * k), 5),
                "mean_query_ms": round(elapsed / qcount, 5), "index_bytes": len(faiss.serialize_index(index)),
                "indexed": len(base), "queries": qcount, "k": k}

    results = [bench("flat-exact", gt)]
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(samples)
    ivf.add(base)
    results.extend(bench("ivf-flat", ivf, min(nlist, p)) for p in nprobes)
    hnsw = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    hnsw.add(base)
    for ef in (16, 64):
        hnsw.hnsw.efSearch = ef
        result = bench("hnsw", hnsw)
        result["ef_search"] = ef
        results.append(result)
    return {"metric": "cosine via L2-normalized inner product", "seed": seed, "nlist": nlist, "results": results,
            "note": "Search-only wall time, not full embedding/network latency; run repeated trials for robust p95."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vectors", required=True, help="Float embeddings .npy with rows aligned to paper IDs")
    p.add_argument("--max-vectors", type=int)
    p.add_argument("--nlist", type=int, default=32)
    p.add_argument("--nprobes", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="artifacts/benchmarks/ann.json")
    args = p.parse_args()
    result = evaluate(load_vectors(args.vectors, max_vectors=args.max_vectors), k=args.k,
                      nlist=args.nlist, nprobes=args.nprobes, queries=args.queries)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
