"""
Fine-tune t5-small to generate a paper's semantic ID directly from a
query, using the synthetic (query, paper_id, semantic_id) triples.
Runs on Modal with a GPU -- t5-small at this data scale trains in
minutes on an A10G vs. hours on CPU. Training logged to Weights &
Biases (loss, eval metrics, GPU utilization) via the Seq2SeqTrainer's
built-in integration.

One-time setup before running:
    modal secret create wandb-secret WANDB_API_KEY=your-key-here

Usage (from your local machine, with `modal` CLI configured):
    modal run eval/train_semantic_id_model.py
"""
import json
from pathlib import Path

import modal

app = modal.App("semantic-id-training")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "datasets", "accelerate", "scikit-learn", "wandb"
)

SYNTHETIC_QUERIES_PATH = Path(__file__).parent / "synthetic_queries.json"
MODEL_NAME = "t5-small"


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(triples: list[dict]):
    import io
    import tarfile

    import numpy as np
    import wandb
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )

    wandb.init(
        project="semantic-id-retrieval",
        config={"model": MODEL_NAME, "epochs": 8, "n_triples": len(triples)},
    )

    # target: semantic ID as literal text, e.g. [3, 1] -> "3 1"
    examples = [
        {"query": t["query"], "target": " ".join(str(x) for x in t["semantic_id"])}
        for t in triples
    ]

    train_examples, val_examples = train_test_split(examples, test_size=0.15, random_state=42)
    print(f"{len(train_examples)} train, {len(val_examples)} val examples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        inputs = tokenizer(
            [f"predict semantic id: {q}" for q in batch["query"]],
            truncation=True, max_length=64, padding="max_length",
        )
        targets = tokenizer(
            batch["target"], truncation=True, max_length=8, padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    train_ds = Dataset.from_list(train_examples).map(preprocess, batched=True)
    val_ds = Dataset.from_list(val_examples).map(preprocess, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/semantic_id_model",
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        logging_steps=20,
        report_to=["wandb"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()

    # exact-match accuracy on held-out val set
    predictions = trainer.predict(val_ds)
    pred_ids = np.where(predictions.predictions != -100, predictions.predictions, tokenizer.pad_token_id)
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    true_texts = [ex["target"] for ex in val_examples]

    exact_matches = sum(p.strip() == t.strip() for p, t in zip(pred_texts, true_texts))
    accuracy = exact_matches / len(true_texts)
    print(f"\nExact-match accuracy on held-out queries: {accuracy:.2%} ({exact_matches}/{len(true_texts)})")
    wandb.log({"held_out_accuracy": accuracy})

    sample_results = [
        {"query": ex["query"], "true": t, "predicted": p}
        for ex, t, p in zip(val_examples[:15], true_texts[:15], pred_texts[:15])
    ]

    model.save_pretrained("/tmp/semantic_id_model/final")
    tokenizer.save_pretrained("/tmp/semantic_id_model/final")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/tmp/semantic_id_model/final", arcname="model")
    buf.seek(0)

    wandb.finish()

    return {"accuracy": accuracy, "samples": sample_results, "model_tarball": buf.getvalue()}


@app.local_entrypoint()
def main():
    triples = json.loads(SYNTHETIC_QUERIES_PATH.read_text())
    print(f"Sending {len(triples)} triples to Modal for training...")

    result = train.remote(triples)

    print(f"\n=== Held-out accuracy: {result['accuracy']:.2%} ===\n")
    print("Sample predictions:")
    for s in result["samples"]:
        marker = "✓" if s["true"] == s["predicted"] else "✗"
        print(f"  {marker} {s['query']!r}")
        print(f"      true={s['true']!r}  predicted={s['predicted']!r}")

    out_path = Path(__file__).parent / "semantic_id_model.tar.gz"
    out_path.write_bytes(result["model_tarball"])
    print(f"\nModel saved -> {out_path}")


if __name__ == "__main__":
    main()