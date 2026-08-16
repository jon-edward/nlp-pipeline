"""
Example 3: SMS spam triage with conditional stages, on a larger dataset
============================================================================

Task: given a mixed inbox of SMS messages, (a) skip near-empty/junk messages
entirely, (b) classify the rest as spam/ham, and (c) only bother extracting
keywords for messages flagged as spam (since that's what a human reviewer
would actually want to skim). This demonstrates `run_if` gating stages
independently -- each stage sees a different slice of the batch.

Data source
-----------
UCI "SMS Spam Collection" dataset: 5,574 real SMS messages labelled spam/ham.

    https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Also mirrored on Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Download and unzip; the file is `SMSSpamCollection`, tab-separated as
`label<TAB>message` with no header row. Point DATA_PATH below at it.

Run
---
    python examples/03_sms_spam_triage_pipeline.py [path/to/SMSSpamCollection]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from nlp_pipeline import (
    ColumnCondition,
    ColumnConditionOp,
    KeywordExtractionStage,
    NLPPipeline,
    TextClassificationStage,
)


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load the real UCI SMS Spam Collection file if given; else fall back
    to the bundled sample.
    """
    if data_path and Path(data_path).exists():
        df = pd.read_csv(
            data_path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="latin-1",
        )
        print(f"Loaded real SMS Spam Collection data: {len(df)} rows from {data_path}.")
        return df

    raise ValueError("Required file SMSSpamCollection not found.")


def main() -> None:
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "SMSSpamCollection"

    df = load_data(data_path)
    df["length"] = df["text"].str.split().str.len()

    train_df = df.sample(frac=0.7, random_state=0)
    classifier = TextClassificationStage(
        text_field="text",
        # only classify messages with at least 3 words -- "ok" / "k" style
        # replies aren't meaningfully spam/ham and would just add noise
        run_if=ColumnCondition("length", ColumnConditionOp.GE, 3),
    ).fit(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
    )

    pipeline = NLPPipeline(
        stages=[
            classifier,
            # only worth extracting keywords for messages already flagged spam
            KeywordExtractionStage(
                text_field="text",
                top_k=5,
                run_if=ColumnCondition("predicted_label", ColumnConditionOp.EQ, "spam"),
            ),
        ]
    )

    result = pipeline(df)

    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(result.df[["label", "text", "predicted_label", "keywords"]])

    skipped = result.df["predicted_label"].isna().sum()
    flagged_spam = (result.df["predicted_label"] == "spam").sum()
    print(f"\n{skipped} short messages skipped by the length filter.")
    print(
        f"{flagged_spam} messages flagged as spam (keywords extracted for those only)."
    )


if __name__ == "__main__":
    main()
