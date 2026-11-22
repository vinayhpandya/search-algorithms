"""Offline fixture tests for the 15K retrieval runner; no AWS required."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scholarsearch.eval_retrieve import BM25, read_queries, retrieve, load_query_vectors
from scholarsearch.embeddings import MODEL
from scholarsearch.eval_retrieve import QUERY_ADAPTER


class RetrievalTests(unittest.TestCase):
    def test_query_tsv(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'q.tsv'
            p.write_text('query_id\tquery\tintent\treviewed\nq001\tmalware analysis\ttechnical_concept\t\n')
            self.assertEqual(read_queries(str(p))[0]['query'], 'malware analysis')

    def test_bm25(self):
        papers = [{'id': 'a', 'title': 'Malware graphs', 'abstract': ''},
                  {'id': 'b', 'title': 'Cloud systems', 'abstract': ''}]
        self.assertEqual(BM25(papers).search('malware', 1)[0][0], 0)

    def test_retrieval(self):
        papers = [{'id':'a', 'title':'Malware detection', 'abstract':'classification'},
                  {'id':'b', 'title':'Networking', 'abstract':'routing'},
                  {'id':'c', 'title':'Malware graphs', 'abstract':'machine learning'}]
        vecs = np.zeros((3, 768), dtype=np.float32)
        vecs[0,0],vecs[1,1],vecs[2,2] = 1,1,1
        qvecs = np.zeros((1,768),dtype=np.float32)
        qvecs[0,0] = 1
        r = retrieve(papers, vecs, [{'query_id':'q1','query':'malware detection'}], qvecs,
                     k=2,candidate_k=3)
        self.assertEqual(r['status'], 'UNJUDGED_RETRIEVAL_ONLY')
        result = r['results'][0]['methods']
        self.assertEqual(result['bm25'][0]['id'],'a')
        self.assertEqual(result['dense'][0]['id'],'a')
        self.assertEqual(result['hybrid'][0]['id'],'a')

    def test_query_vector_id_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'v.jsonl'
            p.write_text(json.dumps({'query_id':'different', 'query':'test','model':MODEL,
                                     'adapter':QUERY_ADAPTER,'vector':[0.] * 768})+'\n')
            with self.assertRaisesRegex(ValueError, 'IDs do not match'):
                load_query_vectors(str(p), [{'query_id':'q1','query':'test'}])


if __name__ == '__main__':
    unittest.main()
