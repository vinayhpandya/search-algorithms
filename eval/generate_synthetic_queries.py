"""
Generate synthetic (query, paper_id, semantic_id) training triples for
the query -> semantic ID generative retrieval model. 40 hand-labeled
eval queries are too few to train a generalizing model, so this
bootstraps a much larger training set: for each paper, an LLM
generates realistic queries someone would type to find that specific
paper, grounded in its actual title+abstract (not just its topic).

Requires OPENAI_API_KEY set in your .env file.
Requires eval/semantic_ids.json to already exist (run
search/build_semantic_ids.py first).

Output: eval/synthetic_queries.json
    [{"query":..., "paper_id":..., "semantic_id":[c, f]}, ...]

Usage:
    python -m eval.generate_synthetic_queries
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, RateLimitError

SEMANTIC_IDS_PATH = Path(__file__).parent / "semantic_ids.json"
PAPERS_PATH = Path(__file__).parent.parent / "data" / "raw" / "papers.json"
OUT_PATH = Path(__file__).parent / "synthetic_queries.json"
MODEL = "gpt-4o-mini"
QUERIES_PER_PAPER = 4

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPT = """You are generating realistic search queries for a paper search \
system over malware-classification research papers.

Given this paper's title and abstract, write {n} short, realistic search \
queries that a researcher would type to find THIS specific paper -- vary \
the phrasing and specificity (some more general, some more precise/technical), \
the way real users phrase queries differently.

Title: {title}
Abstract: {abstract}

Respond with ONLY a JSON object, no other text:
{{"queries": ["query 1", "query 2", "query 3", "query 4"]}}"""


def generate_queries_for_paper(title: str, abstract: str, max_retries: int = 3) -> list[str]:
    prompt = PROMPT.format(n=QUERIES_PER_PAPER, title=title, abstract=abstract)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            return result.get("queries", [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Parse error (attempt {attempt + 1}): {e}")
            time.sleep(1)
        except RateLimitError:
            print("    Rate limited, waiting 20s...")
            time.sleep(20)
    return []


def main():
    semantic_ids = json.loads(SEMANTIC_IDS_PATH.read_text())
    papers = {p["id"]: p for p in json.loads(PAPERS_PATH.read_text())}

    triples = []
    if OUT_PATH.exists():
        triples = json.loads(OUT_PATH.read_text())
    already_done = {t["paper_id"] for t in triples}

    paper_ids = list(semantic_ids.keys())
    print(f"{len(already_done)}/{len(paper_ids)} papers already processed")

    for i, paper_id in enumerate(paper_ids, 1):
        if paper_id in already_done:
            continue

        paper = papers.get(paper_id)
        if paper is None:
            continue

        queries = generate_queries_for_paper(paper["title"], paper["abstract"])
        sem_id = semantic_ids[paper_id]["semantic_id"]

        for q in queries:
            triples.append({"query": q, "paper_id": paper_id, "semantic_id": sem_id})

        if i % 20 == 0 or i == len(paper_ids):
            print(f"[{i}/{len(paper_ids)}] {len(triples)} triples so far -- saving checkpoint")
            OUT_PATH.write_text(json.dumps(triples, indent=2))

    OUT_PATH.write_text(json.dumps(triples, indent=2))
    print(f"\nDone. {len(triples)} synthetic (query, paper_id, semantic_id) triples -> {OUT_PATH}")


if __name__ == "__main__":
    main()