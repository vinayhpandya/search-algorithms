import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scholarsearch import experiment_b as ex
from scholarsearch.storage import join, put_bytes, get_bytes


class ExperimentBTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.query = 'malware detection'
        self.paths = ex.paths(self.root, self.query)

    def tearDown(self):
        self.temp.cleanup()

    def test_isolated_prefix(self):
        self.assertIn('/experiments/experiment-b/', self.paths['checkpoint'])
        self.assertNotEqual(self.paths['root'], self.root)
        with self.assertRaises(ValueError):
            ex.paths(self.root, self.query, '../unsafe')

    def test_ingest_guards_frozen_dataset(self):
        put_bytes(self.paths['freeze'], b'{}')
        with self.assertRaisesRegex(ValueError, 'frozen'):
            ex.ingest(self.root, self.query)

    def test_page_limits(self):
        with self.assertRaisesRegex(ValueError, '100'):
            ex.ingest(self.root, self.query, page_size=200)
        with self.assertRaisesRegex(ValueError, '1-25'):
            ex.ingest(self.root, self.query, pages=100)

    def test_freeze_refuses_too_few_papers(self):
        checkpoint = {'query': self.query, 'page': 1}
        put_bytes(self.paths['checkpoint'], json.dumps(checkpoint).encode())
        with patch.object(ex, 'curate', return_value={'papers': 3}), \
             patch.object(ex, 'load_papers', return_value=[{'id': str(i), 'title': 'x'} for i in range(3)]):
            with self.assertRaisesRegex(ValueError, 'Only 3 unique papers'):
                ex.freeze(self.root, self.query, min_papers=10)
        self.assertFalse(Path(self.paths['freeze']).exists())

    def test_freeze_commits_and_is_idempotent(self):
        put_bytes(self.paths['checkpoint'], json.dumps({'query': self.query, 'page': 1}).encode())
        papers = [{'id': str(i), 'title': 'title'} for i in range(3)]
        with patch.object(ex, 'curate', return_value={'papers': 3}), \
             patch.object(ex, 'load_papers', return_value=papers):
            first = ex.freeze(self.root, self.query, min_papers=3)
        second = ex.freeze(self.root, self.query, min_papers=3)
        self.assertEqual(first, second)
        self.assertEqual(json.loads(get_bytes(self.paths['freeze']))['count'], 3)


if __name__ == '__main__':
    unittest.main()
