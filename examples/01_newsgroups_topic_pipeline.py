"""
Example 1: Topic classification + clustering + keyword extraction
===================================================================

Task: given news posts, (a) classify them into known newsgroup topics,
(b) cluster them to see if the clusters line up with the topics, and
(c) pull out each post's distinguishing keywords -- all as one pipeline.

This is a case where you do NOT want TfidfVectorizeStage in the pipeline:
nothing downstream consumes a shared vector column, so it would just be
dead weight. Classification, clustering, and keyword extraction each fit
their own TF-IDF internally, on purpose (see README.md's "why not combine"
section).

Data source
-----------
The 20 Newsgroups dataset: ~18,000 Usenet posts across 20 topics.
Original source: http://qwone.com/~jason/20Newsgroups/
No manual download needed -- scikit-learn fetches and caches it for you:

    from sklearn.datasets import fetch_20newsgroups
    fetch_20newsgroups(subset="train")

Run
---
    python examples/01_newsgroups_topic_pipeline.py
"""

from __future__ import annotations

import pandas as pd

from nlp_pipeline import (
    KeywordExtractionStage,
    NLPPipeline,
    TextClassificationStage,
    TextClusteringStage,
)

CATEGORIES = ["sci.space", "rec.sport.hockey", "comp.graphics"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df), each with `text` and `label` columns.

    Tries the real 20 Newsgroups dataset first; falls back to a tiny bundled
    sample (so this script always runs, even offline) if the download fails.
    """
    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )
    train_df = pd.DataFrame(
        {"text": train.data, "label": [train.target_names[t] for t in train.target]}
    )
    test_df = pd.DataFrame(
        {"text": test.data, "label": [test.target_names[t] for t in test.target]}
    )
    print(
        f"Loaded real 20 Newsgroups data: {len(train_df)} train / {len(test_df)} test rows."
    )
    return train_df, test_df


def main() -> None:
    train_df, test_df = load_data()

    classifier = TextClassificationStage(text_field="text").fit(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
    )

    pipeline = NLPPipeline(
        stages=[
            KeywordExtractionStage(text_field="text", top_k=5),
            TextClusteringStage(
                text_field="text", n_clusters=len(CATEGORIES), random_state=0
            ),
            classifier,
        ]
    )

    result = pipeline(test_df)

    pd.set_option("display.max_colwidth", 40)
    print(
        result.df[
            [
                "label",
                "predicted_label",
                "predicted_label_confidence",
                "cluster",
                "keywords",
            ]
        ]
    )

    accuracy = (result.df["label"] == result.df["predicted_label"]).mean()
    print(f"\nClassification accuracy on test set: {accuracy:.1%}")


if __name__ == "__main__":
    main()
