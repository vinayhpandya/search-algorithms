"""
Auto-label query-paper relevance pairs using a researcher + critic
LLM pair via OpenAI. The researcher rates each pair independently;
the critic sees the researcher's rating + reasoning and either
confirms or overrides it. Final label = critic's call. Both ratings
are stored so disagreement rate can be measured and reported.

Requires OPENAI_API_KEY set in your environment.

Usage:
    python eval/label_with_llm.py
"""
from dotenv import load_dotenv
load_dotenv()
import json
import os
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

CANDIDATES_PATH = Path(__file__).parent / "candidates.json"
MODEL = "gpt-4o-mini"  # cheap, fast, good enough for this classification task

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

RESEARCHER_PROMPT = """You are judging search relevance for a paper search system \
over malware-classification research papers.

Query: {query}

Paper title: {title}
Paper abstract: {abstract}

Rate how relevant this paper is to the query on this scale:
0 = not relevant (different topic entirely)
1 = somewhat relevant (related area, but not a direct match to what the query asks)
2 = highly relevant (directly addresses what the query is asking about)

Respond with ONLY a JSON object, no other text:
{{"rating": <0, 1, or 2>, "reason": "<one sentence>"}}"""

CRITIC_PROMPT = """You are reviewing a relevance judgment made by another researcher, \
for a paper search system over malware-classification research papers. Be skeptical --
your job is to catch mistakes, not rubber-stamp the researcher's rating.

Query: {query}

Paper title: {title}
Paper abstract: {abstract}

The researcher rated this pair {researcher_rating} ({researcher_reason})

Relevance scale:
0 = not relevant (different topic entirely)
1 = somewhat relevant (related area, but not a direct match to what the query asks)
2 = highly relevant (directly addresses what the query is asking about)

Do you agree with the researcher's rating? If yes, confirm it. If not, give your
own corrected rating. Respond with ONLY a JSON object, no other text:
{{"rating": <0, 1, or 2>, "reason": "<one sentence>", "agreed_with_researcher": <true or false>}}"""


def call_llm(prompt: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"    Parse error (attempt {attempt + 1}): {e}")
            time.sleep(1)
        except RateLimitError:
            print("    Rate limited, waiting 20s...")
            time.sleep(20)
    return None


def judge_pair(query: str, title: str, abstract: str) -> dict:
    researcher = call_llm(RESEARCHER_PROMPT.format(query=query, title=title, abstract=abstract))
    if researcher is None:
        return {"label": 0, "label_source": "llm_failed"}

    critic = call_llm(CRITIC_PROMPT.format(
        query=query, title=title, abstract=abstract,
        researcher_rating=researcher["rating"], researcher_reason=researcher["reason"],
    ))
    if critic is None:
        return {
            "label": researcher["rating"],
            "researcher_rating": researcher["rating"],
            "researcher_reason": researcher["reason"],
            "critic_rating": None,
            "disagreed": False,
            "label_source": "llm_researcher_only",
        }

    disagreed = critic["rating"] != researcher["rating"]
    return {
        "label": critic["rating"],
        "researcher_rating": researcher["rating"],
        "researcher_reason": researcher["reason"],
        "critic_rating": critic["rating"],
        "critic_reason": critic["reason"],
        "disagreed": disagreed,
        "label_source": "llm_researcher_critic",
    }


def main():
    data = json.loads(CANDIDATES_PATH.read_text())
    total = sum(len(v["candidates"]) for v in data.values())
    done = 0
    disagreements = 0

    for query, entry in data.items():
        print(f"\nJudging query: {query!r} ({len(entry['candidates'])} candidates)")
        for candidate in entry["candidates"]:
            if candidate.get("label") is not None:
                done += 1
                continue  # already labeled, don't overwrite

            result = judge_pair(query, candidate["title"], candidate["abstract"])
            candidate.update(result)
            done += 1
            if result.get("disagreed"):
                disagreements += 1
                print(f"  [{done}/{total}] {candidate['title'][:50]}... "
                      f"researcher={result['researcher_rating']} critic={result['critic_rating']} (DISAGREED)")
            else:
                print(f"  [{done}/{total}] {candidate['title'][:50]}... -> {result['label']}")

            # save incrementally so a crash partway through doesn't lose progress
            CANDIDATES_PATH.write_text(json.dumps(data, indent=2))

    print(f"\nDone. Labeled {done}/{total} pairs. Researcher/critic disagreed on {disagreements} pairs.")


if __name__ == "__main__":
    main()