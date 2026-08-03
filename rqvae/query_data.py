"""
Load synthetic (query, semantic_id) pairs for Stage 2 training.

Critically: the train/val split is done by PAPER, not by individual query.
Since each paper contributed ~10 queries all sharing the same target ID, a
random row-level split would leak the same paper (just different phrasings)
into both train and val -- inflating validation accuracy without actually
testing generalization to unseen papers/queries.
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "synthetic_queries.jsonl"


@dataclass
class QueryIDExample:
    query: str
    semantic_id: list[int]
    paper_id: str


def load_examples(path: Path = DEFAULT_PATH) -> list[QueryIDExample]:
    examples = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            examples.append(
                QueryIDExample(
                    query=record["query"],
                    semantic_id=record["semantic_id"],
                    paper_id=record["paper_id"],
                )
            )
    return examples


def split_by_paper(
    examples: list[QueryIDExample], val_fraction: float = 0.15, seed: int = 42
) -> tuple[list[QueryIDExample], list[QueryIDExample]]:
    paper_ids = sorted({ex.paper_id for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(paper_ids)

    n_val_papers = max(1, int(len(paper_ids) * val_fraction))
    val_paper_ids = set(paper_ids[:n_val_papers])

    train = [ex for ex in examples if ex.paper_id not in val_paper_ids]
    val = [ex for ex in examples if ex.paper_id in val_paper_ids]
    return train, val


class QueryIDDataset(Dataset):
    def __init__(self, examples: list[QueryIDExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> QueryIDExample:
        return self.examples[idx]


def collate_fn(batch: list[QueryIDExample]) -> tuple[list[str], torch.Tensor]:
    queries = [ex.query for ex in batch]
    ids = torch.tensor([ex.semantic_id for ex in batch], dtype=torch.long)
    return queries, ids


def build_dataloaders(
    path: Path = DEFAULT_PATH,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    examples = load_examples(path)
    print(f"Loaded {len(examples)} (query, semantic_id) pairs")

    train_examples, val_examples = split_by_paper(examples, val_fraction, seed)
    n_train_papers = len({ex.paper_id for ex in train_examples})
    n_val_papers = len({ex.paper_id for ex in val_examples})
    print(
        f"Split by paper: {len(train_examples)} train queries ({n_train_papers} papers), "
        f"{len(val_examples)} val queries ({n_val_papers} papers)"
    )

    train_loader = DataLoader(
        QueryIDDataset(train_examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        QueryIDDataset(val_examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader