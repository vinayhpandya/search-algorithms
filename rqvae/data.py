"""
Load SPECTER2 embeddings for RQ-VAE training.

Pulls the `embedding` field already backfilled into OpenSearch by
search/embed.py, normalizes them, and exposes a torch Dataset + train/val
split. No text/model inference happens here -- RQ-VAE trains purely on
embeddings that already exist.
"""
from dataclasses import dataclass

import numpy as np
import torch
from opensearchpy import OpenSearch
from torch.utils.data import Dataset, DataLoader, random_split

INDEX_NAME = "papers"


@dataclass
class LoadedEmbeddings:
    paper_ids: list[str]
    embeddings: np.ndarray  # (N, 768), normalized


def fetch_embeddings_from_opensearch(
    client: OpenSearch,
    index_name: str = INDEX_NAME,
    batch_size: int = 500,
    max_docs: int | None = None,
) -> LoadedEmbeddings:
    """Scroll through the index and pull id + embedding for every paper
    that has one. Papers missing an embedding (e.g. embed.py not yet run on
    them) are skipped.

    max_docs: if set, stop after collecting this many documents. Intended
    for smoke tests -- e.g. max_docs=200 to sanity-check the pipeline
    cheaply on CPU before running a real training job."""
    paper_ids: list[str] = []
    vectors: list[list[float]] = []

    query = {"query": {"exists": {"field": "embedding"}}, "_source": ["embedding"]}
    fetch_size = min(batch_size, max_docs) if max_docs else batch_size
    resp = client.search(
        index=index_name, body=query, scroll="2m", size=fetch_size
    )
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]

    while hits:
        for hit in hits:
            paper_ids.append(hit["_id"])
            vectors.append(hit["_source"]["embedding"])
            if max_docs and len(paper_ids) >= max_docs:
                break
        if max_docs and len(paper_ids) >= max_docs:
            break
        resp = client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

    # clear the scroll context on the OpenSearch side so it doesn't linger
    try:
        client.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    embeddings = np.array(vectors, dtype=np.float32)
    return LoadedEmbeddings(paper_ids=paper_ids, embeddings=embeddings)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding to unit length. This matters more here
    than it did for k-means: RQ-VAE reconstruction loss (MSE) is sensitive
    to the raw scale/variance of input dimensions, and un-normalized SPECTER2
    vectors have inconsistent per-dimension scale that can dominate the loss
    and distort what the encoder learns to preserve."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return embeddings / norms


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings).float()

    def __len__(self) -> int:
        return self.embeddings.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


def build_dataloaders(
    embeddings: np.ndarray,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    dataset = EmbeddingDataset(embeddings)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def load_and_prepare(
    client: OpenSearch,
    batch_size: int = 256,
    val_fraction: float = 0.1,
    max_docs: int | None = None,
) -> tuple[LoadedEmbeddings, DataLoader, DataLoader]:
    """Convenience entrypoint: fetch from OpenSearch, normalize, split, wrap in loaders."""
    loaded = fetch_embeddings_from_opensearch(client, max_docs=max_docs)
    print(f"Fetched {len(loaded.paper_ids)} embeddings from OpenSearch")

    normalized = normalize_embeddings(loaded.embeddings)
    loaded = LoadedEmbeddings(paper_ids=loaded.paper_ids, embeddings=normalized)

    train_loader, val_loader = build_dataloaders(
        normalized, batch_size=batch_size, val_fraction=val_fraction
    )
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)} "
        f"(batch_size={batch_size})"
    )
    return loaded, train_loader, val_loader


if __name__ == "__main__":
    client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
    load_and_prepare(client)