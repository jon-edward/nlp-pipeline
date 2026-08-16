"""A stage that extracts each record's top TF-IDF-weighted keywords."""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.base import BaseStage
from nlp_pipeline.stages._sklearn_utils import collect_texts


@dataclass
class KeywordExtractionStage(BaseStage):
    """Extracts the top-N TF-IDF-weighted terms per record, relative to the
    other records seen in the same call.

    Examples
    --------
    >>> stage = KeywordExtractionStage(text_field="text", top_k=1)
    >>> mapping = RecordMapping({
    ...     "a": {"text": "the quick brown fox"},
    ...     "b": {"text": "the lazy dog sleeps"},
    ... })
    >>> stage(mapping)
    >>> "keywords" in mapping["a"]
    True
    """

    text_field: str
    """The record key to read source text from."""

    output_field: str = "keywords"
    """The record key to write the extracted keyword list to."""

    top_k: int = 5
    """Maximum number of keywords to keep per record."""

    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(stop_words="english")
    )
    """The vectorizer used to score terms. Re-fit on the batch each call."""

    def __call__(self, mapping: RecordMapping) -> None:
        keys, texts = collect_texts(mapping, self.text_field)
        if not len(texts):
            return

        matrix = self.vectorizer.fit_transform(texts)
        vocab = self.vectorizer.get_feature_names_out()

        for row_idx, key in enumerate(keys):
            row = matrix.getrow(row_idx)
            ranked = sorted(zip(row.indices, row.data), key=lambda pair: -pair[1])
            top_terms = [vocab[col] for col, _weight in ranked[: self.top_k]]
            mapping[key][self.output_field] = top_terms
