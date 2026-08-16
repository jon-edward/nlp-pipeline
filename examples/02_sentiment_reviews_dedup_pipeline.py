"""
Example 2: Sentiment classification + near-duplicate review detection
=========================================================================

Task: classify product/restaurant reviews as positive or negative, AND
separately flag near-duplicate reviews (e.g. copy-pasted or templated
spam) using cosine similarity over TF-IDF vectors.

This is the case where TfidfVectorizeStage earns its keep as a standalone
stage: the classifier fits and uses its OWN internal vectorizer (as always),
but here something else -- a duplicate-detection pass -- genuinely wants the
raw per-record vectors that TfidfVectorizeStage produces. That's a legitimate
reason to keep the "vectors" column around instead of leaving it unused.

Data source
-----------
UCI "Sentiment Labelled Sentences" dataset (Kotzias et al., 2015):
amazon_cells_labelled.txt, imdb_labelled.txt, yelp_labelled.txt -- ~3,000
sentences total, each tab-separated as `sentence<TAB>label` (1 = positive,
0 = negative).

    https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

Download the zip, extract it, and point DATA_DIR below at the folder
containing the three .txt files. (Also mirrored on Kaggle under the same
name if you prefer a browser download.)

Run
---
    python examples/02_sentiment_reviews_dedup_pipeline.py [DATA_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nlp_pipeline import NLPPipeline, TextClassificationStage, TfidfVectorizeStage

DATA_FILES = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]


def load_data(data_dir: str | Path) -> pd.DataFrame:
    """Load the UCI Sentiment Labelled Sentences files if `data_dir` is given
    and contains them; otherwise fall back to the bundled sample.
    """
    path = Path(data_dir)
    frames = []
    for filename in DATA_FILES:
        file_path = path / filename
        if file_path.exists():
            frames.append(
                pd.read_csv(file_path, sep="\t", header=None, names=["text", "label"])
            )
        else:
            raise ValueError(f"Required file {file_path} not found.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded real UCI sentiment data: {len(df)} rows from {data_dir}.")
    return df


def cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse {term: weight} TF-IDF vectors."""
    shared_terms = vec_a.keys() & vec_b.keys()
    numerator = sum(vec_a[t] * vec_b[t] for t in shared_terms)
    norm_a = np.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = np.sqrt(sum(w * w for w in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def find_near_duplicates(
    df: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[int, int, float]]:
    """Return (row_i, row_j, similarity) for every pair above `threshold`."""
    pairs = []
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_sim(df.iloc[i]["tfidf_vector"], df.iloc[j]["tfidf_vector"])
            if sim >= threshold:
                pairs.append((i, j, sim))
    return pairs


def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    df = load_data(data_dir)

    train_df = df.sample(frac=0.7, random_state=0)
    label_map = {1: "positive", 0: "negative"}

    classifier = TextClassificationStage(text_field="text").fit(
        texts=train_df["text"].tolist(),
        labels=[label_map[label] for label in train_df["label"]],
    )

    pipeline = NLPPipeline(
        stages=[
            # Fit fresh over this whole batch -- used below for dedup, NOT
            # fed into the classifier (which fits its own vectorizer).
            TfidfVectorizeStage(text_field="text", output_field="tfidf_vector"),
            classifier,
        ]
    )

    result = pipeline(df)

    pd.set_option("display.max_colwidth", 50)
    print(result.df[["text", "predicted_label", "predicted_label_confidence"]])

    print("\nNear-duplicate reviews (cosine similarity >= 0.8):")
    duplicates = find_near_duplicates(result.df, threshold=0.8)
    if duplicates:
        for i, j, sim in duplicates:
            print(
                f"  [{sim:.2f}] {result.df.iloc[i]['text']!r}\n         ~= {result.df.iloc[j]['text']!r}"
            )
    else:
        print("  none found")


if __name__ == "__main__":
    main()
