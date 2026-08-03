"""
Generate synthetic (query, semantic_id) pairs for Stage 2 training.

Stage 2 needs a model that maps a *query* to a paper's semantic ID, but we
don't have real user queries -- so for each paper, we ask an LLM to
generate plausible queries a researcher might type to find it, paired with
that paper's semantic_id_rqvae (already backfilled by build_semantic_ids.py).

Design choices worth knowing about:
- 10 queries/paper, deliberately a MIX of close-to-title (easy, keyword-
  overlap) and oblique/natural-language (harder, more realistic) phrasings
  -- so Stage 2 doesn't only learn keyword matching.
- Resumable: writes one JSONL line per paper as it completes, and skips
  papers already present in the output file on restart. With ~194 papers
  x 1 LLM call each (each call returns 10 queries), a transient API
  failure partway through shouldn't mean starting over.

Usage:
    export OPENAI_API_KEY=...
    uv run python -m rqvae.synthetic_queries
    uv run python -m rqvae.synthetic_queries --queries-per-paper 10
"""
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from opensearchpy import OpenSearch

load_dotenv()  # reads .env in the current working directory (and parents) if present

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "synthetic_queries.jsonl"
INDEX_NAME = "papers"
MODEL = "gpt-4o-mini"  # cheap + fast, plenty for short query generation

PROMPT_TEMPLATE = """You are simulating how researchers search for academic papers.

Given the title and abstract of a paper below, generate {n} distinct search \
queries that a researcher might type into a search engine to find this \
paper. Make sure the set is a MIX:
- About half should be close to the paper's own terminology (title/abstract \
keywords, specific method or dataset names).
- About half should be more oblique and natural -- the kind of question or \
loosely-worded query someone would type if they knew roughly what they \
wanted but not the paper's exact terms (e.g. a research question, a \
problem description, a comparison to something else).

Keep each query short (3-12 words), like a real search query, not a full \
sentence question in most cases.

Title: {title}
Abstract: {abstract}

Return ONLY a JSON array of {n} strings, nothing else."""


def fetch_papers_with_semantic_ids(
    client: OpenSearch, index_name: str = INDEX_NAME
) -> list[dict]:
    """Fetch id, title, abstract, and semantic_id_rqvae for every paper that
    has a semantic ID assigned. Papers without one (e.g. build_semantic_ids
    hasn't been run, or was run before this paper was added) are skipped."""
    papers = []
    query = {
        "query": {"exists": {"field": "semantic_id_rqvae"}},
        "_source": ["title", "abstract", "semantic_id_rqvae"],
    }
    resp = client.search(index=index_name, body=query, scroll="2m", size=200)
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]

    while hits:
        for hit in hits:
            src = hit["_source"]
            papers.append(
                {
                    "paper_id": hit["_id"],
                    "title": src.get("title", ""),
                    "abstract": src.get("abstract", ""),
                    "semantic_id": src["semantic_id_rqvae"],
                }
            )
        resp = client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

    try:
        client.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    return papers


def load_already_processed(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    processed = set()
    with open(output_path) as f:
        for line in f:
            try:
                record = json.loads(line)
                processed.add(record["paper_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def generate_queries_for_paper(
    openai_client: OpenAI, title: str, abstract: str, n: int, max_retries: int = 3
) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(n=n, title=title, abstract=abstract[:2000])

    for attempt in range(max_retries):
        try:
            resp = openai_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,  # want variety across the 10 queries
                # not using response_format={"type": "json_object"} here --
                # that mode requires a top-level JSON *object*, but we want
                # a plain JSON array of strings, so we parse manually below
                # and tolerate/strip markdown fences if the model adds them
            )
            text = resp.choices[0].message.content.strip()
            # strip markdown code fences if the model added them despite instructions
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            queries = json.loads(text)
            if isinstance(queries, list) and len(queries) > 0:
                return queries[:n]
        except (json.JSONDecodeError, Exception) as e:
            wait = 2**attempt
            print(f"  Retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait}s)")
            time.sleep(wait)

    print(f"  Failed to generate queries for '{title[:60]}' after {max_retries} retries")
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-per-paper", type=int, default=10)
    parser.add_argument("--opensearch-host", type=str, default="localhost")
    parser.add_argument("--opensearch-port", type=int, default=9200)
    parser.add_argument("--index", type=str, default=INDEX_NAME)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os_client = OpenSearch(hosts=[{"host": args.opensearch_host, "port": args.opensearch_port}])
    openai_client = OpenAI()  # reads OPENAI_API_KEY from environment

    papers = fetch_papers_with_semantic_ids(os_client, args.index)
    print(f"Fetched {len(papers)} papers with semantic IDs assigned")

    already_processed = load_already_processed(output_path)
    if already_processed:
        print(f"Resuming: {len(already_processed)} papers already processed, skipping those")

    remaining = [p for p in papers if p["paper_id"] not in already_processed]
    print(f"Generating queries for {len(remaining)} remaining papers "
          f"({args.queries_per_paper} queries each, model={MODEL})")

    with open(output_path, "a") as f:
        for i, paper in enumerate(remaining):
            queries = generate_queries_for_paper(
                openai_client, paper["title"], paper["abstract"], args.queries_per_paper
            )
            for q in queries:
                record = {
                    "paper_id": paper["paper_id"],
                    "query": q,
                    "semantic_id": paper["semantic_id"],
                }
                f.write(json.dumps(record) + "\n")
            f.flush()  # so a crash mid-run doesn't lose completed papers

            if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
                print(f"  [{i + 1}/{len(remaining)}] {paper['title'][:60]} -> {len(queries)} queries")

    total_pairs = sum(1 for _ in open(output_path))
    print(f"\nDone. {total_pairs} total (query, semantic_id) pairs written to {output_path}")


if __name__ == "__main__":
    main()