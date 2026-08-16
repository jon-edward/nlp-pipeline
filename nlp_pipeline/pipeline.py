"""The top-level orchestrator that runs a sequence of stages over a dataframe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, MutableMapping

import pandas as pd

from nlp_pipeline.stages.base import BaseStage
from nlp_pipeline.conditions import NoopCondition, Condition
from nlp_pipeline.records import RecordMapping, RecordMappingSlice, DictRecord
from nlp_pipeline.uuid_gen import default_uuid_generator


@dataclass
class PipelineRunResult:
    """The output of a :meth:`NLPPipeline.__call__` call."""

    df: pd.DataFrame
    """
    The input dataframe after modification from all stages. The output dataframe does not
    necessarily follow the same order as the input dataframe.
    """


@dataclass
class NLPPipeline:
    """Orchestrates a sequence of :class:`~nlp_pipeline.stages.base.BaseStage`
    over a dataframe.

    Add stages via ``pipeline.stages.append(...)``, then call :meth:`~NLPPipeline.__call__`
    to run the pipeline.

    Examples
    --------
    >>> pipeline = NLPPipeline(stages=[my_qa_stage, my_classifier_stage]) # doctest: +SKIP
    >>> result = pipeline(df) # doctest: +SKIP
    >>> result.df.head() # doctest: +SKIP
    """

    stages: list[BaseStage | Callable[[RecordMapping], None]] = field(
        default_factory=list
    )
    """
    The ordered list of stages to run. Each stage can be a function taking a :class:`~nlp_pipeline.records.RecordMapping` 
    or a :class:`~nlp_pipeline.stages.base.BaseStage`.
    """

    def __call__(self, df: pd.DataFrame) -> PipelineRunResult:
        """Run every stage, in order, over ``df``.

        Parameters
        ----------
        df:
            The input dataframe.

        Returns
        -------
        PipelineRunResult
            The output dataframe.
        """

        uuid_gen = default_uuid_generator()
        uuid_gen.reset()  # Reset the UUID generator at each run

        records: MutableMapping[str, DictRecord] = {
            uuid_gen.next(): record_data for record_data in df.to_dict(orient="records")
        }

        for stage in self.stages:
            # Handle callable stages
            run_if: Condition = getattr(stage, "run_if", NoopCondition())

            mapping = RecordMapping(records)

            if not isinstance(run_if, NoopCondition):
                keys = (key for key, value in records.items() if run_if.matches(value))
                mapping = RecordMappingSlice(mapping, keys)

            stage(mapping)

        df = pd.DataFrame.from_records(tuple(records.values()))
        return PipelineRunResult(df=df)
