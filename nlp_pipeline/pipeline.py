"""The top-level orchestrator that runs a sequence of stages over a dataframe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import uuid

import pandas as pd
from tqdm import tqdm

from nlp_pipeline.stages.base import BaseStage


@dataclass
class PipelineRunResult:
    """The output of an :meth:`NLPPipeline.run` call.

    Attributes
    ----------
    df:
        The input dataframe with each stage's output columns added, indexed
        the same way as the input.
    """

    df: pd.DataFrame


_ID_PROBE_MAX_TRIES = 100


@dataclass
class NLPPipeline:
    """Orchestrates a sequence of :class:`~nlp_pipeline.stages.base.BaseStage`
    over a dataframe.

    Add stages via ``pipeline.stages.append(...)``, then call :meth:`run`.
    Each stage's :meth:`~nlp_pipeline.stages.base.BaseStage.fit` is called
    once against the current batch of records before that stage's
    :meth:`~nlp_pipeline.stages.base.BaseStage.run` is applied per-record, so
    stateful stages (classifiers, cluster models) can train on the full
    batch before predicting.

    Attributes
    ----------
    stages:
        The ordered list of stages to run. Each stage sees the columns added
        by all prior stages.

    Examples
    --------
    >>> pipeline = NLPPipeline(stages=[my_qa_stage, my_classifier_stage])
    >>> result = pipeline(df)
    >>> result.df.head()
    """

    stages: Sequence[BaseStage] = field(default_factory=list)

    def __call__(self, df: pd.DataFrame) -> PipelineRunResult:
        """Run every stage, in order, over ``df``.

        Parameters
        ----------
        df:
            The input dataframe.

        Returns
        -------
        PipelineRunResult
            Wraps the resulting dataframe with all stages' output columns.
        """

        records = {
            uuid.uuid4(): record_data for record_data in df.to_dict(orient="records")
        }

        for stage in self.stages:
            records = stage(records)

        df = pd.DataFrame.from_records(records)
        df.drop(columns=[id_column], inplace=True, errors="ignore")
        return PipelineRunResult(df=df)
