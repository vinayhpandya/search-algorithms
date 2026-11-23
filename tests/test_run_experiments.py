"""Offline tests for the experiment runner: synthetic corpus, no AWS, no models."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scholarsearch.embeddings import MODEL
from scholarsearch.eval_retrieve import QUERY_ADAPTER
from scholarsearch import run_experiments as rx

DIGEST = 'test-digest'


def fake_corpus(n=400, dim=768, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    words = ['malware', 'graph', 'android', 'network', 'vision', 'language']
    papers = [{'id': f'p{i}', 'title': f'{words[i % 6]} study {i}', 'abstract': words[(i + 1) % 6]}
              for i in range(n)]
    return papers, vecs


class RunnerFixture:
    """Shared temp dir, fake corpus and query files (mixin: holds no tests)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.papers, self.vectors = fake_corpus()
        queries = [('q1', 'malware graph'), ('q2', 'android network')]
        (d / 'q.tsv').write_text('query_id\tquery\n' + ''.join(f'{a}\t{b}\n' for a, b in queries))
        with (d / 'qv.jsonl').open('w') as f:
            for i, (qid, q) in enumerate(queries):
                f.write(json.dumps({'query_id': qid, 'query': q, 'model': MODEL,
                                    'adapter': QUERY_ADAPTER, 'vector': self.vectors[i].tolist()}) + '\n')
        self.labels = d / 'labels.jsonl'
        self.labels.write_text('')
        self.config = {
            'name': 'unit', 'root': 'unused', 'query': 'malware detection', 'out_root': str(d / 'runs'),
            'ann': {'queries': 20, 'nlist': 4, 'nprobes': [1, 4], 'ef_search': [16], 'pq_m': 8,
                    'pq_bits': 4, 'repeats': 1},
            'retrieval': {'queries': str(d / 'q.tsv'), 'query_vectors': str(d / 'qv.jsonl'),
                          'k': 3, 'candidate_k': 5},
            'report': {'labels': str(self.labels)},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def loader(self, root, query, *, need_papers):
        return (self.papers if need_papers else None), self.vectors, {'source_digest': DIGEST, 'model': MODEL}

    def write_labels(self, retrieval, *, skip=0):
        pairs = sorted({(r['query_id'], h['id']) for r in retrieval['results']
                        for hits in r['methods'].values() for h in hits[:retrieval['k']]})
        digest = retrieval['corpus']['source_digest']
        with self.labels.open('w') as f:
            for i, (q, p) in enumerate(pairs[skip:]):
                f.write(json.dumps({'query_id': q, 'paper_id': p, 'source_digest': digest,
                                    'label': i % 3}) + '\n')


class RunnerTests(RunnerFixture, unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            rx.validate_config({**self.config, 'name': 'bad name'})
        with self.assertRaises(ValueError):
            rx.validate_config({**self.config, 'ann': {'typo_setting': 1}})
        with self.assertRaises(ValueError):
            rx.enabled_stages(self.config, ['nope'])
        self.assertEqual(rx.enabled_stages(self.config, ['report', 'ann']), ['ann', 'report'])

    def test_run_id_is_deterministic_per_config(self):
        t = datetime(2026, 11, 22, tzinfo=timezone.utc)
        self.assertEqual(rx.make_run_id(self.config, t), rx.make_run_id(dict(self.config), t))
        self.assertNotEqual(rx.make_run_id(self.config, t),
                            rx.make_run_id({**self.config, 'query': 'x'}, t))

    def test_full_run_without_labels_reports_gaps_not_scores(self):
        m = rx.run(self.config, loader=self.loader)
        run_dir = Path(m['run_dir'])
        self.assertTrue((run_dir / 'ann.json').exists())
        self.assertTrue((run_dir / 'retrieval.json').exists())
        report = json.loads((run_dir / 'report.json').read_text())
        self.assertEqual(report['status'], 'INCOMPLETE_UNJUDGED_PAIRS')
        self.assertGreater(report['coverage']['missing'], 0)
        self.assertNotIn('metrics', report)
        saved = json.loads((run_dir / 'run_manifest.json').read_text())
        self.assertEqual(saved['results']['ann']['status'], 'complete')
        methods = {r['method'] for r in json.loads((run_dir / 'ann.json').read_text())['results']}
        self.assertTrue({'exact-flat', 'hnsw'} <= methods)

    def test_report_from_earlier_run_with_complete_labels(self):
        first = rx.run(self.config, stages=['retrieval'], loader=self.loader)
        retrieval = json.loads((Path(first['run_dir']) / 'retrieval.json').read_text())
        self.write_labels(retrieval)
        second = rx.run(self.config, stages=['report'], from_run=first['run_dir'],
                        loader=self.loader, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
        report = json.loads((Path(second['run_dir']) / 'report.json').read_text())
        self.assertEqual(report['status'], 'COMPLETE_LLM_JUDGED_POOLED')
        self.assertEqual(report['coverage']['missing'], 0)
        self.assertEqual(set(report['metrics']), {'bm25', 'dense', 'hybrid'})

    def test_partial_labels_counted_per_method(self):
        first = rx.run(self.config, stages=['retrieval'], loader=self.loader)
        retrieval = json.loads((Path(first['run_dir']) / 'retrieval.json').read_text())
        self.write_labels(retrieval, skip=2)
        cov = rx.label_coverage(retrieval, {(json.loads(l)['query_id'], json.loads(l)['paper_id']): 1
                                            for l in self.labels.read_text().splitlines()})
        self.assertEqual(cov['missing'], 2)
        self.assertGreaterEqual(sum(cov['missing_by_method'].values()), 2)

    def test_report_alone_requires_retrieval_source(self):
        with self.assertRaises(ValueError):
            rx.run(self.config, stages=['report'], loader=self.loader)

    def test_failed_stage_is_recorded(self):
        bad = {**self.config, 'retrieval': {**self.config['retrieval'], 'query_vectors': '/missing'}}
        with self.assertRaises(FileNotFoundError):
            rx.run(bad, loader=self.loader)
        run_dir = next(Path(self.config['out_root']).iterdir())
        saved = json.loads((run_dir / 'run_manifest.json').read_text())
        self.assertEqual(saved['results']['ann']['status'], 'complete')
        self.assertEqual(saved['results']['retrieval']['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
