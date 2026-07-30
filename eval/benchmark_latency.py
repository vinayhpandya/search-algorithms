"""
Benchmark per-query latency across all 6 search methods. Runs each
eval query N times (discarding the first as cold-start warmup) and
reports mean/p50/p95 latency per method.

Usage:
    python -m eval.benchmark_latency
"""
import statistics
import time

from search.bm25 import search_bm25
from search.dense import search_dense
from search.hybrid import search_hybrid
from search.rerank import search_rerank
from search.search_late_interaction import search_late_interaction
from search.search_semantic_id import search_semantic_id
from eval.queries import EVAL_QUERIES

# Note: LTR isn't benchmarked here directly -- it operates on
# precomputed features (BM25/dense scores etc.), not a live query path,
# so "latency" for it would mean something different (feature lookup +
# a fast model.predict() call) rather than a comparable end-to-end search.
METHODS = {
    "bm25": search_bm25,
    "dense": search_dense,
    "hybrid": search_hybrid,
    "rerank": search_rerank,
    "late_interaction": search_late_interaction,
    "semantic_id": search_semantic_id,
}

RUNS_PER_QUERY = 5  # first run discarded as warmup
WARMUP_QUERIES = EVAL_QUERIES[:2]  # queries used just to warm up models before timing


def warmup():
    print("Warming up models (loading into memory, not timed)...")
    for fn in METHODS.values():
        for q in WARMUP_QUERIES:
            fn(q, k=10)


def benchmark_method(name: str, fn) -> dict:
    latencies = []
    for query in EVAL_QUERIES:
        for run in range(RUNS_PER_QUERY):
            start = time.perf_counter()
            fn(query, k=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if run == 0:
                continue  # discard first run per query as warmup
            latencies.append(elapsed_ms)
    return {
        "mean": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
        "min": min(latencies),
        "max": max(latencies),
    }


def main():
    warmup()

    results = {}
    for name, fn in METHODS.items():
        print(f"\nBenchmarking {name}...")
        results[name] = benchmark_method(name, fn)
        print(f"  mean={results[name]['mean']:.1f}ms  p50={results[name]['p50']:.1f}ms  p95={results[name]['p95']:.1f}ms")

    print(f"\n{'Method':<18} {'Mean (ms)':>12} {'P50 (ms)':>12} {'P95 (ms)':>12}")
    print("-" * 56)
    for name, stats in results.items():
        print(f"{name:<18} {stats['mean']:>12.1f} {stats['p50']:>12.1f} {stats['p95']:>12.1f}")


if __name__ == "__main__":
    main()