"""
Enrich data/raw/papers.json in-place with citation counts from
Semantic Scholar. Separated from collect_papers.py so this step can
be rerun independently (e.g. after a partial failure) without
re-hitting arXiv.

Usage:
    python data/enrich_citations.py
"""
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

S2_API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
PAPERS_PATH = Path(__file__).parent / "raw" / "papers.json"


def fetch_citation_count(arxiv_id: str, max_retries: int = 3) -> int:
    """Best-effort lookup: returns 0 on any failure rather than
    aborting the whole run over one bad/missing paper."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                S2_API.format(arxiv_id=arxiv_id),
                params={"fields": "citationCount"},
                timeout=10,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                time.sleep(wait)
                continue
            if resp.ok:
                return resp.json().get("citationCount", 0) or 0
            return 0
        except requests.RequestException:
            return 0
    return 0


def main():
    papers = json.loads(PAPERS_PATH.read_text())

    for p in tqdm(papers, desc="Fetching citation counts"):
        p["citation_count"] = fetch_citation_count(p["id"])
        time.sleep(0.34)  # stay under S2's unauthenticated rate limit (~3 req/s)

    PAPERS_PATH.write_text(json.dumps(papers, indent=2))
    print(f"Updated {len(papers)} papers with citation counts -> {PAPERS_PATH}")


if __name__ == "__main__":
    main()