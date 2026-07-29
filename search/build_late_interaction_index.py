"""
Build a late-interaction (ColBERT) index in Qdrant, using fastembed's
LateInteractionTextEmbedding -- lighter dependency footprint than
RAGatouille/PyLate (no LangChain, minimal sentence-transformers
coupling), avoiding the version conflicts hit with those two.

Usage:
    python search/build_qdrant_index.py
"""
import json
from pathlib import Path

from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

PAPERS_PATH = Path(__file__).parent.parent / "data" / "raw" / "papers.json"
COLLECTION_NAME = "papers_colbert"
MODEL_NAME = "colbert-ir/colbertv2.0"

client = QdrantClient(url="http://localhost:6333")


def main():
    papers = json.loads(PAPERS_PATH.read_text())
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]

    print(f"Loading {MODEL_NAME}...")
    model = LateInteractionTextEmbedding(MODEL_NAME)

    # fastembed's ColBERT wrapper returns embedding dim via its own property
    embedding_dim = model.embedding_size
    print(f"Embedding dimension: {embedding_dim}")

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists, deleting to recreate...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    )
    print(f"Created collection '{COLLECTION_NAME}'")

    print(f"Encoding {len(texts)} papers (this takes a while)...")
    batch_size = 16
    point_id = 0
    for i in range(0, len(texts), batch_size):
        batch_papers = papers[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = list(model.embed(batch_texts))  # list of per-doc token-vector arrays

        points = []
        for paper, embedding in zip(batch_papers, batch_embeddings):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),  # shape: (num_tokens, embedding_dim)
                    payload={
                        "arxiv_id": paper["id"],
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "year": paper["year"],
                        "citation_count": paper.get("citation_count", 0),
                    },
                )
            )
            point_id += 1

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Indexed {min(i + batch_size, len(texts))}/{len(texts)}")

    print(f"Done. {point_id} papers indexed into Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()