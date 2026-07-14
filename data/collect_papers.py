"""
Collect malware-classification-related papers from arXiv, then enrich
each with a citation count from Semantic Scholar.

Output: data/raw/papers.json
    A list of dicts: {id, title, abstract, authors, year, categories, citation_count}

Why arXiv for the corpus:
    Free, no auth required, good coverage of ML/security research.
Why Semantic Scholar for citations:
    arXiv itself doesn't expose citation counts. We need citation count
    later as a relevance-proxy / LTR feature (Day 2).

Usage:
    python data/collect_arxiv.py --max-results 2000
"""
import argparse
import json
import time
from pathlib import Path

import feedparser
import requests
from tqdm import tqdm

ARXIV_API = "http://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"

# Queries chosen to cover the main malware-classification subareas:
# static/dynamic analysis, opcode/API-call features, deep learning
# classifiers, and adversarial evasion -- so the eval queries later
# (Day 2) have real variety to test against.
SEARCH_QUERIES = [
    "malware classification",
    "malware detection deep learning",
    "malware family classification",
    "opcode sequence malware",
    "API call malware detection",
    "adversarial malware evasion",
    "ransomware detection machine learning",
    "malware image classification CNN",
]

OUT_PATH = Path(__file__).parent / "raw" / "papers.json"


def fetch_arxiv(query: str, max_results: int) -> list[dict]:
    """Fetch paper metadata from the arXiv API for a single query."""
    params = {
        "search_query": f'cat:cs.CR AND abs:"{query}"',
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1].split("v")[0]  # strip version
        papers.append(
            {
                "id": arxiv_id,
                "title": " ".join(entry.title.split()),
                "abstract": " ".join(entry.summary.split()),
                "authors": [a.name for a in entry.authors],
                "year": int(entry.published[:4]),
                "categories": [t["term"] for t in entry.tags],
            }
        )
    return papers


def enrich_with_citations(papers: list[dict]) -> list[dict]:
    """Add a citation_count field via Semantic Scholar. Best-effort:
    missing/failed lookups default to 0 rather than aborting the run."""
    for p in tqdm(papers, desc="Fetching citation counts"):
        try:
            resp = requests.get(
                S2_API.format(arxiv_id=p["id"]),
                params={"fields": "citationCount"},
                timeout=10,
            )
            if resp.ok:
                p["citation_count"] = resp.json().get("citationCount", 0) or 0
            else:
                p["citation_count"] = 0
        except requests.RequestException:
            p["citation_count"] = 0
        time.sleep(0.34)  # stay under S2's unauthenticated rate limit (~3 req/s)
    return papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=300,
                         help="max results PER query (there are 8 queries)")
    parser.add_argument("--skip-citations", action="store_true",
                         help="skip the slow Semantic Scholar enrichment step")
    args = parser.parse_args()

    all_papers = {}
    for q in SEARCH_QUERIES:
        print(f"Fetching: {q!r}")
        for p in fetch_arxiv(q, args.max_results):
            all_papers[p["id"]] = p  # dedupe across overlapping queries
        time.sleep(3)  # be polite to arXiv's API

    papers = list(all_papers.values())
    print(f"Collected {len(papers)} unique papers")

    if not args.skip_citations:
        papers = enrich_with_citations(papers)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(papers, indent=2))
    print(f"Wrote {len(papers)} papers to {OUT_PATH}")


if __name__ == "__main__":
    main()