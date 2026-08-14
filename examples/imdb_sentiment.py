"""
An NLPpipeline that uses a simple logistic classifier to predict sentiment over IMDB reviews.
"""

from dataclasses import dataclass
import pathlib
from typing import Sequence

from nlp_pipeline.conditions import FunctionCondition
from nlp_pipeline.stages.logistic_classifier import LogisticClassifier
from nlp_pipeline.stages.base import BaseStage
from nlp_pipeline import NLPPipeline
from records import RunRecord, DictRecord

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def imdb_dataset():
    return pd.read_csv(PROJECT_ROOT / "data" / "IMDB Dataset.csv")


@dataclass
class TrimNoInference(BaseStage):
    """Custom stage that trims records that have no inference."""
    cols: Sequence[str]

    def process_records(self, run_records: Sequence[RunRecord]) -> Sequence[DictRecord]:
        out_records = [
            record.record_data
            for record in run_records
            if record.do_run and any(record.record_data[col] for col in self.cols)
        ]
        print(f"Trimmed {len(run_records) - len(out_records)} records")
        return out_records


pipeline = NLPPipeline(
    stages=[
        LogisticClassifier(
            input_column="review",
            label_column="sentiment",
            output_column="model-sentiment-bad",
            description="logistic classifier for reviews containing 'bad'",
            run_if=FunctionCondition(lambda r: "bad" in r["review"]),
        ),
        LogisticClassifier(
            input_column="review",
            label_column="sentiment",
            output_column="model-sentiment-great",
            description="logistic classifier for reviews containing 'great'",
            run_if=FunctionCondition(lambda r: "great" in r["review"]),
        ),
        TrimNoInference(cols=["model-sentiment-bad", "model-sentiment-great"]),
    ]
)


pipeline(imdb_dataset()).df.to_csv(PROJECT_ROOT / "data" / "output.csv", index=False)
