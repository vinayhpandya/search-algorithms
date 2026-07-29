"""
Train a LightGBM LTR model (LambdaMART) on the feature table from
build_ltr_features.py. Split is by QUERY, not by pair -- keeping all
of a query's pairs together on one side avoids leaking signal across
train/test (see README for the small-data caveat: ~28 train queries
is a small regime for LTR, results should be read as directional).

Usage:
    python -m eval.train_ltr
"""
import json
import random
from pathlib import Path

import lightgbm as lgb
import pandas as pd

FEATURES_PATH = Path(__file__).parent / "ltr_features.json"
MODEL_PATH = Path(__file__).parent / "ltr_model.txt"
TEST_PREDICTIONS_PATH = Path(__file__).parent / "ltr_test_predictions.json"

FEATURE_COLS = ["bm25_score", "dense_score", "citation_count", "year", "title_exact_match"]
TEST_FRACTION = 0.3
SEED = 42


def main():
    rows = json.loads(FEATURES_PATH.read_text())
    df = pd.DataFrame(rows)

    queries = sorted(df["query"].unique())
    random.seed(SEED)
    random.shuffle(queries)
    n_test = max(1, int(len(queries) * TEST_FRACTION))
    test_queries = set(queries[:n_test])
    train_queries = set(queries[n_test:])
    print(f"{len(train_queries)} train queries, {len(test_queries)} test queries")

    train_df = df[df["query"].isin(train_queries)].sort_values("query")
    test_df = df[df["query"].isin(test_queries)].sort_values("query")

    # LightGBM's ranker needs "group" sizes: how many rows belong to
    # each query, in the same contiguous order as the sorted dataframe.
    train_groups = train_df.groupby("query", sort=False).size().tolist()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,       # kept small -- few training queries, avoid overfitting
        min_child_samples=5,
    )
    model.fit(
        train_df[FEATURE_COLS],
        train_df["label"],
        group=train_groups,
    )
    model.booster_.save_model(str(MODEL_PATH))
    print(f"Model saved -> {MODEL_PATH}")

    # Predict on held-out test queries, save per-query ranked order
    test_df = test_df.copy()
    test_df["ltr_score"] = model.predict(test_df[FEATURE_COLS])

    predictions = {}
    for query, group in test_df.groupby("query"):
        ranked = group.sort_values("ltr_score", ascending=False)
        predictions[query] = ranked["id"].tolist()

    TEST_PREDICTIONS_PATH.write_text(json.dumps(predictions, indent=2))
    print(f"Test predictions saved -> {TEST_PREDICTIONS_PATH} ({len(predictions)} queries)")


if __name__ == "__main__":
    main()