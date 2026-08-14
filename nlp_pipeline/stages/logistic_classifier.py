"""
Classify text using a Logistic Regression classifier.
"""

from dataclasses import dataclass, field
from typing import Sequence

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from records import DictRecord, RunRecord
from nlp_pipeline.stages.base import BaseStage


@dataclass
class LogisticClassifier(BaseStage):
    """
    Classify text using a Logistic Regression classifier.
    """

    input_column: str
    """The column containing the text to classify."""

    label_column: str
    """The column containing the labels."""

    output_column: str
    """The column to store the predictions in."""

    test_split: float = field(default=0.2, kw_only=True)
    """The fraction of the data to use for testing."""

    print_eval: bool = field(default=True, kw_only=True)
    """Whether to print evaluation metrics."""

    pipeline: Pipeline | None = field(init=False, default=None)
    """The trained classifier pipeline."""

    _predictions: dict[int, object] = field(init=False, default_factory=dict)
    """The predictions for each record. Internal use only."""

    @staticmethod
    def _build_vectorizer():
        return TfidfVectorizer(
            max_df=0.95, min_df=2, max_features=10_000, stop_words="english"
        )

    @staticmethod
    def _build_classifier():
        return LogisticRegression()

    def _build_pipeline(self):
        return Pipeline(
            [("vec", self._build_vectorizer()), ("clf", self._build_classifier())]
        )

    def fit(self, run_records: Sequence[RunRecord]) -> None:
        eligible: list[RunRecord] = [r for r in run_records if r.do_run]
        self._predictions = {}

        if not eligible:
            self.pipeline = None
            return

        texts = [r.record_data[self.input_column] for r in eligible]
        labels = [r.record_data[self.label_column] for r in eligible]

        train_idx, test_idx = train_test_split(
            range(len(texts)), test_size=self.test_split
        )

        X_train, y_train = [texts[i] for i in train_idx], [labels[i] for i in train_idx]
        X_test, y_test = [texts[i] for i in test_idx], [labels[i] for i in test_idx]

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)

        y_pred_test = self.pipeline.predict(X_test)

        if self.print_eval:
            print(
                f"=== Classification Report{'' if self.description is None else ' (' + self.description + ')'}: ==="
            )
            print(classification_report(y_test, y_pred_test))

        y_pred_train = self.pipeline.predict(X_train)

        for i, pred in zip(train_idx, y_pred_train):
            self._predictions[eligible[i].record_data[self.id_column]] = pred
        for i, pred in zip(test_idx, y_pred_test):
            self._predictions[eligible[i].record_data[self.id_column]] = pred

    def process_records(self, run_records: Sequence[RunRecord]) -> Sequence[DictRecord]:
        for record in run_records:
            record.record_data[self.output_column] = (
                self._predictions.get(record.record_data[self.id_column])
                if record.do_run
                else None
            )
        return [record.record_data for record in run_records]
