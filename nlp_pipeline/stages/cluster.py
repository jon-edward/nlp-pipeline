"""A stage that groups similar text together using scikit-learn's KMeans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.base import BaseStage
from nlp_pipeline.stages._sklearn_utils import collect_texts


@dataclass
class TextClusteringStage(BaseStage):
    """Clusters each batch of records' text with TF-IDF features + KMeans.

    Clustering, unlike classification, has no fixed label set to learn ahead of
    time, so this stage fits a fresh :class:`~sklearn.cluster.KMeans` model over
    whatever text is visible on each call (i.e. the current batch, or whatever
    ``run_if`` lets through). Cluster ids are therefore only comparable *within*
    a single pipeline run, not across separate runs.

    Examples
    --------
    >>> stage = TextClusteringStage(text_field="text", n_clusters=2, random_state=0)
    >>> mapping = RecordMapping({
    ...     "a": {"text": "cats and dogs are pets"},
    ...     "b": {"text": "dogs and cats are pets"},
    ...     "c": {"text": "stocks and bonds are investments"},
    ...     "d": {"text": "bonds and stocks are investments"},
    ... })
    >>> stage(mapping)
    >>> mapping["a"]["cluster"] == mapping["b"]["cluster"]
    True
    """

    text_field: str
    """The record key to read source text from."""

    output_field: str = "cluster"
    """The record key to write the assigned cluster id to."""

    n_clusters: int = 2
    """Number of clusters to fit. Automatically capped at the number of records
    in the batch, since KMeans cannot fit more clusters than samples.
    """

    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(stop_words="english")
    )
    """The vectorizer used to turn text into features before clustering."""

    random_state: int | None = 0
    """Random state forwarded to :class:`~sklearn.cluster.KMeans` for reproducibility."""

    n_init: int | Literal["auto", "warn"] = "auto"
    """Forwarded to :class:`~sklearn.cluster.KMeans`."""

    def __call__(self, mapping: RecordMapping) -> None:
        keys, texts = collect_texts(mapping, self.text_field)
        if not len(texts):
            return

        n_clusters = max(1, min(self.n_clusters, len(texts)))

        features = self.vectorizer.fit_transform(texts)
        model = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=self.n_init
        )
        labels = model.fit_predict(features)

        for key, cluster_id in zip(keys, labels):
            mapping[key][self.output_field] = int(cluster_id)
