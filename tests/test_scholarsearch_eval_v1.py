import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from scholarsearch.evaluation_v1 import (
    freeze_queries, openalex_id, parse_run, score_runs, read_qrels,
    main, read_jsonl, load_json, read_tsv, write_tsv, write_json, write_jsonl,
    sha256_file, annotate_pool, judgments,
)


class EvaluationV1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = {'schema':'scholarsearch.corpus.v1','source_digest':'test-only',
                         'documents':3,'paper_ids':['W1','W2','W3']}

    def test_demo_runnable(self):
        output = self.root/'toy'
        main(['demo','--out',str(output)])
        report = load_json(output/'toy_report.json')
        self.assertTrue(report['TOY_DATA_DO_NOT_PRESENT_AS_REAL'])
        self.assertEqual(report['metrics']['bm25_toy']['status'],'COMPLETE_POOLED_JUDGMENTS')
        self.assertGreater(report['metrics']['dense_toy']['ndcg_at_k'],report['metrics']['bm25_toy']['ndcg_at_k'])
        main(['validate-run','--manifest',str(output/'corpus_manifest.json'),
              '--queries',str(output/'queries.jsonl'),
              '--runs',str(output/'runs/bm25_toy.trec'),str(output/'runs/dense_toy.trec')])
        main(['score','--manifest',str(output/'corpus_manifest.json'),
              '--queries',str(output/'queries.jsonl'),
              '--runs',str(output/'runs/bm25_toy.trec'),str(output/'runs/dense_toy.trec'),
              '--qrels',str(output/'qrels.tsv'),'--k','3','--out',str(output/'real_score_path_test.json')])

    def test_queries_require_explicit_review(self):
        rows = [{'query_id':'q001','query':'valid query for research','reviewed':'no'},
                {'query_id':'q002','query':'a different query phrase','reviewed':'yes'}]
        with self.assertRaisesRegex(ValueError,'reviewed'):
            freeze_queries(rows,manifest=self.manifest,dev=1)
        rows[0]['reviewed']='yes'
        frozen = freeze_queries(rows,manifest=self.manifest,dev=1)
        self.assertEqual({x['split'] for x in frozen},{'dev','test'})
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            freeze_queries(rows+[rows[0]],manifest=self.manifest,dev=1)

    def test_run_rejects_wrong_corpus(self):
        f = self.root/'run.trec'
        f.write_text('q001 Q0 W999 1 1.0 bm25\n')
        with self.assertRaisesRegex(ValueError,'not in frozen corpus'):
            parse_run(f,{'q001'},set(self.manifest['paper_ids']))
        f.write_text('q001 Q0 W1 1 1.0 bm25\nq001 Q0 W1 2 0.9 bm25\n')
        with self.assertRaisesRegex(ValueError,'duplicate'):
            parse_run(f,{'q001'},set(self.manifest['paper_ids']))

    def test_score_refuses_unjudged(self):
        queries=[{'query_id':'q001','split':'test'},{'query_id':'q002','split':'test'}]
        runs={'bm25':{'q001':['W1','W2'],'q002':['W1','W3']}}
        qrels={('q001','W1'):3,('q001','W2'):0,('q002','W1'):2}
        m=score_runs(runs,queries,qrels,split='test',k=2)['bm25']
        self.assertEqual(m['status'],'INCOMPLETE_JUDGMENTS_NO_AGGREGATE_SCORE')
        self.assertIsNone(m['ndcg_at_k'])
        self.assertEqual(m['queries_covered'],1)
        qrels[('q002','W3')]=0
        m=score_runs(runs,queries,qrels,split='test',k=2)['bm25']
        self.assertEqual(m['status'],'COMPLETE_POOLED_JUDGMENTS')
        self.assertAlmostEqual(m['ndcg_at_k'],1.)

    def test_judgments_validate_provenance(self):
        toy = self.root/'toy'
        main(['demo','--out',str(toy)])
        rows = [{'query_id':'q001','paper_id':'W1','relevance':'2','judge':'reviewer-a'},
                {'query_id':'q002','paper_id':'W2','relevance':'0','judge':'reviewer-a'}]
        sheet = toy/'review.tsv'
        write_tsv(sheet,['query_id','paper_id','relevance','judge'],rows)
        main(['judgments','--manifest',str(toy/'corpus_manifest.json'),
              '--queries',str(toy/'queries.jsonl'), '--reviewed',str(sheet),
              '--out',str(toy/'qrels-export.tsv')])
        self.assertEqual(len(read_qrels(toy/'qrels-export.tsv')),2)
        rows[0]['judge']=''
        write_tsv(sheet,['query_id','paper_id','relevance','judge'],rows)
        with self.assertRaisesRegex(ValueError,'judge identifier'):
            main(['judgments','--manifest',str(toy/'corpus_manifest.json'),
                  '--queries',str(toy/'queries.jsonl'), '--reviewed',str(sheet),
                  '--out',str(toy/'qrels-export.tsv')])

    def test_pool_annotation_requires_exact_ids(self):
        toy=self.root/'toy'
        main(['demo','--out',str(toy)])
        main(['pool','--manifest',str(toy/'corpus_manifest.json'),
              '--queries',str(toy/'queries.jsonl'),
              '--runs',str(toy/'runs/bm25_toy.trec'),
              '--depth','2','--allow-id-only','--out',str(toy/'pool.tsv')])
        write_jsonl(toy/'papers.jsonl',[{'id':f'W{i}','title':f'Paper {i}','abstract':'Abstract'} for i in range(1,7)])
        main(['annotate-pool','--manifest',str(toy/'corpus_manifest.json'),
              '--queries',str(toy/'queries.jsonl'), '--pool',str(toy/'pool.tsv'),
              '--corpus-jsonl',str(toy/'papers.jsonl'), '--out',str(toy/'review.tsv')])
        enriched=read_tsv(toy/'review.tsv')
        self.assertTrue(all(x['title'] for x in enriched))
        self.assertTrue(all('run' not in x for x in enriched))


if __name__ == '__main__':
    unittest.main()
