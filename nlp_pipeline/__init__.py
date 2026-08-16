"""
nlp_pipeline: a small orchestration layer for composing NLP stages
(QA, classification, clustering) over a pandas DataFrame.

See NLPPipeline for the main entry point.
"""

from nlp_pipeline.conditions import (
    Condition,
    ColumnCondition,
    ColumnConditionOp,
    ConditionGroup,
    ConditionGroupOp,
    FunctionCondition,
    NoopCondition,
)
from nlp_pipeline.pipeline import NLPPipeline, PipelineRunResult
from nlp_pipeline.records import DictRecord
from nlp_pipeline.stages import (
    BaseStage,
    KeywordExtractionStage,
    TextClassificationStage,
    TextClusteringStage,
    TfidfVectorizeStage,
)

__all__ = [
    "Condition",
    "ColumnCondition",
    "ColumnConditionOp",
    "ConditionGroup",
    "ConditionGroupOp",
    "DictRecord",
    "FunctionCondition",
    "NoopCondition",
    "NLPPipeline",
    "PipelineRunResult",
    "BaseStage",
    "KeywordExtractionStage",
    "TextClassificationStage",
    "TextClusteringStage",
    "TfidfVectorizeStage",
]
