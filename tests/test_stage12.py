import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scholarsearch.ingest import run
from scholarsearch.curate import curate, inverted_abstract


class StageTests(unittest.TestCase):
    def test_reconstruct_abstract(self):
        self.assertEqual(inverted_abstract({"Malware": [0], "detection": [1]}), "Malware detection")

    def test_resume_dedup_and_curate(self):
        with tempfile.TemporaryDirectory() as path:
            page1 = {"results": [{"id": "https://openalex.org/W1", "display_name": "One",
                                   "abstract_inverted_index": {"A": [0], "paper": [1]}}],
                     "meta": {"next_cursor": "cursor2"}}
            page2 = {"results": [{"id": "https://openalex.org/W1", "display_name": "One updated"},
                                  {"id": "https://openalex.org/W2", "display_name": "Two"}],
                     "meta": {"next_cursor": None}}
            client = Mock()
            client.get.side_effect = [Mock(status_code=200, json=lambda: page1),
                                      Mock(status_code=200, json=lambda: page2)]
            state = run(root=path, query="malware", max_pages=1, session=client, sleep=lambda _: None)
            self.assertEqual(state["page"], 1)
            self.assertFalse(state["done"])
            state = run(root=path, query="malware", max_pages=2, session=client, sleep=lambda _: None)
            self.assertEqual(state["page"], 2)
            self.assertTrue(state["done"])
            out = curate(path, "malware")
            self.assertEqual(out["papers"], 2)
            rows = [json.loads(x) for x in Path(out["output"]).read_text().splitlines()]
            self.assertEqual(rows[0]["title"], "One updated")
            self.assertEqual(client.get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
