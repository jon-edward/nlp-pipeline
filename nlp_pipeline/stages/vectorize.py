"""A stage that embeds text into TF-IDF vectors using scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.base import BaseStage
from nlp_pipeline.stages._sklearn_utils import collect_texts


@dataclass
class TfidfVectorizeStage(BaseStage):
    """Fits a :class:`~sklearn.feature_extraction.text.TfidfVectorizer` on the
    text seen in each call and writes the resulting vector back onto every
    record.

    Because :class:`~nlp_pipeline.pipeline.NLPPipeline` runs each stage over
    the whole (or a conditionally-filtered) batch of records at once, this
    stage fits and transforms in the same step -- there is no separate
    ``fit``/``predict`` split. If you need a vectorizer trained ahead of time
    and reused across runs, construct one yourself and pass it in via
    ``vectorizer``, then set ``refit=False``.

    Examples
    --------
    >>> stage = TfidfVectorizeStage(text_field="text", output_field="tfidf")
    >>> mapping = RecordMapping({"a": {"text": "cats and dogs"}, "b": {"text": "dogs and birds"}})
    >>> stage(mapping)
    >>> sorted(mapping["a"]["tfidf"].keys())  # doctest: +SKIP
    ['and', 'birds', 'cats', 'dogs']
    """

    text_field: str
    """The record key to read source text from."""

    output_field: str = "tfidf_vector"
    """The record key to write the resulting sparse-as-dict vector to."""

    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(stop_words="english")
    )
    """The vectorizer to use. Reused (and re-fit, unless ``refit=False``) across calls."""

    refit: bool = True
    """Whether to re-fit ``vectorizer`` on each call using the current batch of text.

    Set this to ``False`` if ``vectorizer`` was already fit on an external corpus and
    the pipeline should only ever call ``.transform``.
    """

    as_dict: bool = True
    """If ``True``, store a ``{term: weight}`` dict of nonzero weights per record
    (readable, easy to assert on in tests). If ``False``, store the dense
    ``list[float]`` row instead.
    """

    def __call__(self, mapping: RecordMapping) -> None:
        keys, texts = collect_texts(mapping, self.text_field)
        if not len(texts):
            return

        if self.refit:
            matrix = self.vectorizer.fit_transform(texts)
        else:
            matrix = self.vectorizer.transform(texts)

        vocab = self.vectorizer.get_feature_names_out()

        for row_idx, key in enumerate(keys):
            row = matrix.getrow(row_idx)
            if self.as_dict:
                value = {
                    vocab[col]: float(weight)
                    for col, weight in zip(row.indices, row.data)
                }
            else:
                value = row.toarray().ravel().tolist()
            mapping[key][self.output_field] = value
