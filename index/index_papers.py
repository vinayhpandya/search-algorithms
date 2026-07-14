"""
Bulk-load data/raw/papers.json into the 'papers' OpenSearch index.
Embedding field is left empty here -- populated in the next step.

Usage:
    python index/index_papers.py
"""
import json
from pathlib import Path

from opensearchpy import OpenSearch, helpers

INDEX_NAME = "papers"
PAPERS_PATH = Path(__file__).parent.parent / "data" / "raw" / "papers.json"

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

def gen_actions(papers):
    for p in papers:
        yield {
            "_index": INDEX_NAME,
            "_id": p["id"],
            "_source": {
                "id": p["id"],
                "title": p["title"],
                "abstract": p["abstract"],
                "authors": p["authors"],
                "year": p["year"],
                "categories": p["categories"],
                "citation_count": p.get("citation_count", 0),
            },
        }

def main():
    papers = json.loads(PAPERS_PATH.read_text())
    success, errors = helpers.bulk(client, gen_actions(papers), raise_on_error=False)
    print(f"Indexed {success} papers, {len(errors)} errors")
    if errors:
        print("First error:", errors[0])

if __name__ == "__main__":
    main()