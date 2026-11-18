"""PR6 cheap offline validations, plus Faiss integration if installed."""
import unittest

import numpy as np

from scholarsearch.ann_benchmark import (validate_vectors, split_vectors,
                                         recall_at_k, percentile_ms, run_benchmark)


class TestPR6(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(90, 16)).astype(np.float32)
        self.vectors = data / np.linalg.norm(data, axis=1, keepdims=True)

    def test_validates_normalized_float32(self):
        validate_vectors(self.vectors, expected_count=90, dimension=16)
        with self.assertRaises(ValueError):
            validate_vectors((self.vectors * 2).copy())
        with self.assertRaises(ValueError):
            validate_vectors(self.vectors.astype(np.float64))
        with self.assertRaises(ValueError):
            validate_vectors(self.vectors, expected_count=91)

    def test_deterministic_disjoint_split(self):
        b1, q1 = split_vectors(self.vectors, seed=42, queries=10)
        b2, q2 = split_vectors(self.vectors, seed=42, queries=10)
        np.testing.assert_array_equal(b1, b2)
        np.testing.assert_array_equal(q1, q2)
        self.assertEqual(len(b1), 80)
        self.assertEqual(len(q1), 10)
        self.assertFalse(any(np.array_equal(v, w) for v in q1 for w in b1))

    def test_recall_missing_results(self):
        actual = np.array([[3, 1, -1], [7, 8, 9]])
        truth = np.array([[1, 2, 3], [7, 10, 11]])
        self.assertAlmostEqual(recall_at_k(actual, truth), 3 / 6)

    def test_latency_percentiles(self):
        res = percentile_ms([1.0, 2.0, 3.0])
        self.assertEqual(res['p50_ms'], 2.0)
        self.assertGreater(res['p95_ms'], 2.0)

    def test_faiss_smoke_if_available(self):
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest('faiss-cpu not installed in this test interpreter')
        result = run_benchmark(self.vectors, query_count=10, k=5, nlist=4,
                               pq_m=4, repeats=1, warmups=0)
        names = [r['method'] for r in result['results']]
        self.assertIn('exact-flat', names)
        self.assertIn('hnsw', names)
        self.assertIn('ivf-pq', [r['method'] for r in result['skipped']])
        self.assertEqual(result['results'][0]['recall_at_k'], 1.0)


if __name__ == '__main__':
    unittest.main()
