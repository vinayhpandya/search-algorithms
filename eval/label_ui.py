"""
Local web UI for reviewing query-paper relevance labels in
eval/candidates.json.

Modes:
    /              unlabeled pairs only (fallback for anything the LLM missed)
    /?all=1        every pair, for browsing/overriding any label
    /?sample=1     a fixed stratified random sample for spot-checking --
                    weighted toward researcher/critic disagreements, since
                    those are the most informative pairs to human-check

Usage:
    python eval/label_ui.py
    (then open http://localhost:5050)
"""
import json
import random
from pathlib import Path

from flask import Flask, request, jsonify

CANDIDATES_PATH = Path(__file__).parent / "candidates.json"
SAMPLE_PATH = Path(__file__).parent / "spotcheck_sample.json"
SAMPLE_SIZE = 65
DISAGREEMENT_SHARE = 0.5  # roughly half the sample prioritizes disagreement pairs

app = Flask(__name__)


def load_data():
    return json.loads(CANDIDATES_PATH.read_text())


def save_data(data):
    CANDIDATES_PATH.write_text(json.dumps(data, indent=2))


def all_pairs(data):
    pairs = []
    for query, entry in data.items():
        for idx, candidate in enumerate(entry["candidates"]):
            pairs.append((query, idx, candidate))
    return pairs


def build_or_load_sample(data):
    """Generate the stratified sample once and persist it, so repeated
    page loads walk through the same fixed list rather than resampling."""
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text())

    pairs = all_pairs(data)
    disagreed = [(q, i) for q, i, c in pairs if c.get("disagreed")]
    others = [(q, i) for q, i, c in pairs if not c.get("disagreed") and c.get("label") is not None]

    n_disagree = min(len(disagreed), int(SAMPLE_SIZE * DISAGREEMENT_SHARE))
    n_other = min(len(others), SAMPLE_SIZE - n_disagree)

    random.seed(42)  # reproducible sample
    sample = random.sample(disagreed, n_disagree) + random.sample(others, n_other)
    random.shuffle(sample)

    sample_list = [{"query": q, "idx": i} for q, i in sample]
    SAMPLE_PATH.write_text(json.dumps(sample_list, indent=2))
    return sample_list


def flat_pairs(data, mode: str):
    if mode == "sample":
        sample = build_or_load_sample(data)
        pairs = []
        for item in sample:
            entry = data[item["query"]]["candidates"][item["idx"]]
            if entry.get("label_source") != "human_spotcheck":  # not yet reviewed
                pairs.append((item["query"], item["idx"], entry))
        return pairs
    if mode == "all":
        return all_pairs(data)
    # default: unlabeled only
    return [(q, i, c) for q, i, c in all_pairs(data) if c.get("label") is None]


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Relevance Labeling</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }}
  .progress {{ color: #666; margin-bottom: 20px; }}
  .query {{ font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  .title {{ font-size: 20px; font-weight: 600; margin: 8px 0 12px; }}
  .abstract {{ line-height: 1.5; color: #333; margin-bottom: 24px; }}
  .meta {{ font-size: 13px; color: #999; margin-bottom: 20px; background: #f7f7f7; padding: 10px; border-radius: 6px; }}
  .buttons {{ display: flex; gap: 12px; }}
  button {{ flex: 1; padding: 16px; font-size: 16px; border: 2px solid #ddd; border-radius: 8px; background: white; cursor: pointer; }}
  button:hover {{ background: #f5f5f5; }}
  .b0:hover {{ border-color: #e57373; }}
  .b1:hover {{ border-color: #ffb74d; }}
  .b2:hover {{ border-color: #81c784; }}
  .done {{ text-align: center; padding: 60px 0; color: #888; }}
  .nav {{ margin-top: 20px; text-align: center; }}
  .nav a {{ color: #888; text-decoration: none; margin: 0 10px; }}
</style>
</head>
<body>
{body}
<script>
function label(query, idx, value, mode) {{
  fetch('/label', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{query: query, idx: idx, label: value, mode: mode}})
  }}).then(() => location.reload());
}}
</script>
</body>
</html>
"""


def get_mode():
    if request.args.get("sample") == "1":
        return "sample"
    if request.args.get("all") == "1":
        return "all"
    return "unlabeled"


@app.route("/")
def index():
    mode = get_mode()
    data = load_data()
    pairs = flat_pairs(data, mode)

    if not pairs:
        body = f'<div class="done">Nothing left in this view. 🎉<br><br>' \
               f'<a href="/?sample=1">Spot-check sample</a> · ' \
               f'<a href="/?all=1">Review all</a></div>'
        return PAGE_TEMPLATE.format(body=body)

    query, idx, candidate = pairs[0]
    remaining = len(pairs)

    meta = ""
    if candidate.get("label_source", "").startswith("llm"):
        meta = (f'<div class="meta">LLM label: {candidate.get("label")} '
                f'(researcher={candidate.get("researcher_rating")}, '
                f'critic={candidate.get("critic_rating")}'
                f'{" — DISAGREED" if candidate.get("disagreed") else ""})<br>'
                f'{candidate.get("critic_reason", candidate.get("researcher_reason", ""))}</div>')

    mode_label = {"sample": "spot-check sample", "all": "reviewing all", "unlabeled": "unlabeled only"}[mode]
    body = f"""
    <div class="progress">{remaining} pair(s) remaining ({mode_label})</div>
    <div class="query">Query: {query}</div>
    <div class="title">{candidate['title']}</div>
    <div class="abstract">{candidate['abstract']}</div>
    {meta}
    <div class="buttons">
      <button class="b0" onclick="label('{escape_js(query)}', {idx}, 0, '{mode}')">0 — Not relevant</button>
      <button class="b1" onclick="label('{escape_js(query)}', {idx}, 1, '{mode}')">1 — Somewhat relevant</button>
      <button class="b2" onclick="label('{escape_js(query)}', {idx}, 2, '{mode}')">2 — Highly relevant</button>
    </div>
    <div class="nav">
      <a href="/?sample=1">Spot-check sample</a> ·
      <a href="/?all=1">Review all</a> ·
      <a href="/">Unlabeled only</a>
    </div>
    """
    return PAGE_TEMPLATE.format(body=body)


def escape_js(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


@app.route("/label", methods=["POST"])
def set_label():
    payload = request.get_json()
    data = load_data()
    candidate = data[payload["query"]]["candidates"][payload["idx"]]

    if payload["mode"] == "sample":
        # preserve the original LLM label for agreement scoring, record human's separately
        candidate["llm_label"] = candidate.get("label")
        candidate["human_label"] = payload["label"]
        candidate["label"] = payload["label"]
        candidate["label_source"] = "human_spotcheck"
    else:
        candidate["label"] = payload["label"]
        candidate["label_source"] = "human"

    save_data(data)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(port=5050, debug=True)