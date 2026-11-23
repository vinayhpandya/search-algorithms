"""Offline tests for new retrieval methods and the capped judge stage. No AWS, models or API."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scholarsearch import run_experiments as rx
from scholarsearch.eval_retrieve import retrieve
from scholarsearch.methods import parse_methods, run_methods
from tests.test_run_experiments import RunnerFixture, fake_corpus

QUERIES = [{'query_id': 'q1', 'query': 'malware graph'}, {'query_id': 'q2', 'query': 'android network'}]


class FakeReranker:
    """Scores papers by the integer in their id, so the expected order is known."""
    def __init__(self, model):
        self.model, self.prepared = model, set()

    def prepare(self, hits):
        self.prepared |= {h['id'] for h in hits}

    def score(self, query, hits):
        assert {h['id'] for h in hits} <= self.prepared
        return [float(h['id'][1:]) for h in hits]


FAKES = {'cross_encoder': FakeReranker, 'colbert': FakeReranker}


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.papers, self.vectors = fake_corpus(n=300)
        self.qvecs = np.ascontiguousarray(self.vectors[:2])

    def test_parse_methods(self):
        self.assertEqual(parse_methods(['bm25', 'hybrid+colbert']),
                         (['bm25', 'hybrid'], [('hybrid', 'colbert')]))
        for bad in (['nope'], ['hybrid+nope'], ['bm25', 'bm25']):
            with self.assertRaises(ValueError):
                parse_methods(bad)

    def test_baseline_methods_identical_to_eval_retrieve(self):
        old = retrieve(self.papers, self.vectors, QUERIES, self.qvecs, k=10, candidate_k=50)
        new = run_methods(self.papers, self.vectors, QUERIES, self.qvecs,
                          methods=['bm25', 'dense', 'hybrid'], k=10, candidate_k=50, pool=100)
        for a, b in zip(old['results'], new['results']):
            for m in ('bm25', 'dense', 'hybrid'):
                self.assertEqual([h['id'] for h in a['methods'][m]], [h['id'] for h in b['methods'][m]], m)
            self.assertIn('hybrid_total', b['latency_ms'])

    def test_hnsw_matches_exact_with_large_ef(self):
        r = run_methods(self.papers, self.vectors, QUERIES, self.qvecs,
                        methods=['dense', 'dense_hnsw', 'hybrid_hnsw'], k=10,
                        hnsw={'ef_search': 300})
        for row in r['results']:
            self.assertEqual({h['id'] for h in row['methods']['dense']},
                             {h['id'] for h in row['methods']['dense_hnsw']})
        self.assertIn('hnsw_build_ms', r['setup'])

    def test_reranker_reorders_pool_and_adds_latency(self):
        r = run_methods(self.papers, self.vectors, QUERIES, self.qvecs, k=5, pool=20,
                        methods=['hybrid', 'hybrid+cross_encoder', 'hybrid_hnsw+colbert'],
                        reranker_factories=FAKES)
        self.assertEqual(r['methods'], ['hybrid', 'hybrid+cross_encoder', 'hybrid_hnsw+colbert'])
        for row in r['results']:
            ranked = row['methods']['hybrid+cross_encoder']
            ids = [int(h['id'][1:]) for h in ranked]
            self.assertEqual(ids, sorted(ids, reverse=True))
            self.assertEqual(len(ranked), 5)
            self.assertTrue(all(1 <= h['base_rank'] <= 20 for h in ranked))
            self.assertGreaterEqual(row['latency_ms']['hybrid+cross_encoder'], row['latency_ms']['hybrid'])
            self.assertEqual(set(row['methods']), set(r['methods']))


class JudgeStageTests(RunnerFixture, unittest.TestCase):
    """Reuses the runner fixture (temp dir, fake corpus, query files)."""

    def setUp(self):
        super().setUp()
        self.config['retrieval']['methods'] = ['bm25', 'dense', 'hybrid', 'hybrid+colbert']
        self.config['judge'] = {'enabled': True, 'model': 'fake-model', 'max_usd': 25,
                                'price_per_million_input': 0.15, 'price_per_million_output': 0.60}
        self.calls = 0

    def fake_call(self, pair):
        self.calls += 1
        return {'label': int(pair['paper_id'][1:]) % 3, 'reason': 'test',
                'usage': {'prompt_tokens': 300, 'completion_tokens': 40}}

    def go(self, **kw):
        return rx.run(self.config, loader=self.loader, reranker_factories=FAKES,
                      judge_call=self.fake_call, **kw)

    def outputs(self, manifest):
        d = Path(manifest['run_dir'])
        return {n: json.loads((d / f'{n}.json').read_text()) for n in ('judge', 'report')}

    def test_judge_then_report_scores_every_method(self):
        out = self.outputs(self.go(stages=['retrieval', 'judge', 'report']))
        self.assertEqual(out['judge']['status'], 'complete')
        self.assertEqual(out['judge']['still_missing'], 0)
        self.assertGreater(out['judge']['usd_this_run'], 0)
        self.assertEqual(out['report']['status'], 'COMPLETE_LLM_JUDGED_POOLED')
        self.assertEqual(set(out['report']['metrics']), {'bm25', 'dense', 'hybrid', 'hybrid+colbert'})
        self.assertIn('retrieval_p50_ms', out['report']['metrics']['hybrid+colbert'])
        ledger = json.loads((self.labels.parent / 'judge_spend.json').read_text())
        self.assertEqual(len(ledger['runs']), 1)

    def test_second_run_reuses_labels_and_spends_nothing(self):
        self.go(stages=['retrieval', 'judge'])
        first = self.calls
        out = self.outputs(self.go(stages=['retrieval', 'judge', 'report'],
                                   now=rx.datetime(2031, 1, 1, tzinfo=rx.timezone.utc)))
        self.assertEqual(self.calls, first)
        self.assertEqual(out['judge']['pairs_judged'], 0)
        self.assertEqual(out['report']['status'], 'COMPLETE_LLM_JUDGED_POOLED')

    def test_cap_is_cumulative_and_partial_labels_are_kept(self):
        # Each fake pair costs ~$0.000069; the runner projects >= $0.0005 per next pair,
        # so a $0.0006 cap allows exactly 2 pairs, and nothing on the next run.
        self.config['judge']['max_usd'] = 0.0006
        out = self.outputs(self.go(stages=['retrieval', 'judge', 'report']))
        self.assertEqual(out['judge']['status'], 'STOPPED_AT_BUDGET')
        self.assertEqual(out['judge']['pairs_judged'], 2)
        self.assertLessEqual(out['judge']['usd_total'], 0.0006)
        self.assertGreater(out['judge']['still_missing'], 0)
        self.assertEqual(len(self.labels.read_text().splitlines()), 2)  # partial labels persisted
        self.assertEqual(out['report']['status'], 'INCOMPLETE_UNJUDGED_PAIRS')
        calls = self.calls
        second = self.go(stages=['retrieval', 'judge'],
                         now=rx.datetime(2031, 1, 1, tzinfo=rx.timezone.utc))
        self.assertEqual(self.calls, calls)  # cap is cumulative across runs
        judge2 = json.loads((Path(second['run_dir']) / 'judge.json').read_text())
        self.assertEqual(judge2['status'], 'STOPPED_AT_BUDGET')

    def test_judge_is_opt_in_and_needs_key(self):
        self.config['judge']['enabled'] = False
        self.assertNotIn('judge', rx.enabled_stages(self.config, None))
        self.config['judge']['enabled'] = True
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                rx.run(self.config, stages=['retrieval', 'judge'], loader=self.loader,
                       reranker_factories=FAKES)

    def test_prices_required(self):
        del self.config['judge']['price_per_million_input']
        with self.assertRaises(ValueError):
            rx.validate_config(self.config)


if __name__ == '__main__':
    unittest.main()
