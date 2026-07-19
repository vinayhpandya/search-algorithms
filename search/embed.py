"""
Compute SPECTER2 embeddings for every paper and backfill them into
the 'embedding' field of the existing OpenSearch index.

Usage:
    python search/embed.py
"""
import json
from pathlib import Path

import torch
from opensearchpy import OpenSearch, helpers
from transformers import AutoTokenizer, AutoModel

INDEX_NAME = "papers"
MODEL_NAME = "allenai/specter2_base"
PAPERS_PATH = Path(__file__).parent.parent / "data" / "raw" / "papers.json"
BATCH_SIZE = 16

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_batch(texts: list[str], tokenizer, model) -> list[list[float]]:
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    outputs = model(**inputs)
    # SPECTER2 uses the [CLS] token embedding as the document representation
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.tolist()


def gen_update_actions(papers, tokenizer, model):
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i : i + BATCH_SIZE]
        # SPECTER2's expected input format: title + [SEP] + abstract
        texts = [f"{p['title']}{tokenizer.sep_token}{p['abstract']}" for p in batch]
        vectors = embed_batch(texts, tokenizer, model)
        for p, vec in zip(batch, vectors):
            yield {
                "_op_type": "update",
                "_index": INDEX_NAME,
                "_id": p["id"],
                "doc": {"embedding": vec},
            }


def main():
    papers = json.loads(PAPERS_PATH.read_text())
    print(f"Loading {MODEL_NAME}...")
    tokenizer, model = load_model()

    print(f"Embedding + updating {len(papers)} papers...")
    success, errors = helpers.bulk(
        client, gen_update_actions(papers, tokenizer, model), raise_on_error=False
    )
    print(f"Updated {success} papers, {len(errors)} errors")
    if errors:
        print("First error:", errors[0])


if __name__ == "__main__":
    main()