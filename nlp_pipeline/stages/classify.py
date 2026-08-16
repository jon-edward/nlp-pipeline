"""A stage that classifies text using a scikit-learn estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.base import BaseStage
from nlp_pipeline.stages._sklearn_utils import collect_texts


def _default_model() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(stop_words="english"),
        LogisticRegression(max_iter=1000),
    )


@dataclass
class TextClassificationStage(BaseStage):
    """Predicts a label for each record's text using a scikit-learn classifier.

    Unlike :class:`~nlp_pipeline.stages.vectorize.TfidfVectorizeStage`, this stage
    is *not* fit as part of running the pipeline: classification only makes sense
    against a model trained ahead of time, so call :meth:`fit` before adding this
    stage to a :class:`~nlp_pipeline.pipeline.NLPPipeline`.

    Examples
    --------
    >>> stage = TextClassificationStage(text_field="text")
    >>> _ = stage.fit(["great product", "terrible service"], ["positive", "negative"])
    >>> mapping = RecordMapping({"a": {"text": "great service"}})
    >>> stage(mapping)
    >>> mapping["a"]["predicted_label"] in {"positive", "negative"}
    True
    """

    text_field: str
    """The record key to read source text from."""

    output_field: str = "predicted_label"
    """The record key to write the predicted label to."""

    output_proba_field: str | None = "predicted_label_confidence"
    """The record key to write the winning class's predicted probability to.
    Set to ``None`` to skip (e.g. for estimators without ``predict_proba``).
    """

    model: Pipeline = field(default_factory=_default_model)
    """The scikit-learn estimator (typically a :class:`~sklearn.pipeline.Pipeline`
    ending in a classifier) used for prediction. Must implement ``predict``, and
    ``predict_proba`` if ``output_proba_field`` is set.
    """

    _fitted: bool = field(default=False, init=False, repr=False, compare=False)

    def fit(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> "TextClassificationStage":
        """Fit ``model`` on labeled examples. Must be called before use in a pipeline."""
        self.model.fit(list(texts), list(labels))
        self._fitted = True
        return self

    def __call__(self, mapping: RecordMapping) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} was added to a pipeline without calling "
                "`.fit(texts, labels)` first."
            )

        keys, texts = collect_texts(mapping, self.text_field)
        if not len(texts):
            return

        predictions = self.model.predict(texts)

        probabilities = None
        if self.output_proba_field is not None and hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(texts).max(axis=1)

        for row_idx, key in enumerate(keys):
            record = mapping[key]
            record[self.output_field] = predictions[row_idx]
            if probabilities is not None:
                record[self.output_proba_field] = float(probabilities[row_idx])
