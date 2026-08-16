import pandas as pd
import pytest

from nlp_pipeline.conditions import ColumnCondition, ColumnConditionOp
from nlp_pipeline.pipeline import NLPPipeline, PipelineRunResult
from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.base import BaseStage


class UppercaseStage(BaseStage):
    """Simple test stage: uppercases a `text` field in place."""

    def __call__(self, mapping: RecordMapping) -> None:
        for key in list(mapping.keys()):
            mapping[key]["text"] = mapping[key]["text"].upper()


class AddColumnStage(BaseStage):
    """Simple test stage: stamps a constant value onto every record it sees."""

    def __init__(self, field_name: str, value, **kwargs):
        super().__init__(**kwargs)
        self.field_name = field_name
        self.value = value

    def __call__(self, mapping: RecordMapping) -> None:
        for key in list(mapping.keys()):
            mapping[key][self.field_name] = self.value


class RecordingFunctionStage:
    """Plain callable (non-BaseStage) stage used to test the function-stage path."""

    def __init__(self):
        self.seen_keys = []

    def __call__(self, mapping: RecordMapping) -> None:
        self.seen_keys = list(mapping.keys())


def test_empty_pipeline_returns_equivalent_dataframe():
    df = pd.DataFrame({"text": ["a", "b"]})
    pipeline = NLPPipeline(stages=[])
    result = pipeline(df)

    assert isinstance(result, PipelineRunResult)
    assert sorted(result.df["text"].tolist()) == ["a", "b"]


def test_single_stage_runs_and_mutates_records():
    df = pd.DataFrame({"text": ["hello", "world"]})
    pipeline = NLPPipeline(stages=[UppercaseStage()])
    result = pipeline(df)

    assert sorted(result.df["text"].tolist()) == ["HELLO", "WORLD"]


def test_multiple_stages_run_in_order():
    df = pd.DataFrame({"text": ["hello"]})
    pipeline = NLPPipeline(
        stages=[
            UppercaseStage(),
            AddColumnStage("shouted", True),
        ]
    )
    result = pipeline(df)

    assert result.df.iloc[0]["text"] == "HELLO"
    assert bool(result.df.iloc[0]["shouted"]) is True


def test_plain_callable_stage_is_supported():
    df = pd.DataFrame({"text": ["a", "b", "c"]})
    fn_stage = RecordingFunctionStage()
    pipeline = NLPPipeline(stages=[fn_stage])
    pipeline(df)

    assert len(fn_stage.seen_keys) == 3


def test_run_if_condition_filters_which_records_a_stage_sees():
    df = pd.DataFrame({"text": ["keep", "drop"], "flag": [True, False]})
    stage = AddColumnStage(
        "touched",
        True,
        run_if=ColumnCondition("flag", ColumnConditionOp.EQ, True),
    )
    pipeline = NLPPipeline(stages=[stage])
    result = pipeline(df)

    touched_by_text = dict(zip(result.df["text"], result.df.get("touched")))
    assert touched_by_text["keep"] is True
    # "drop" was filtered out by run_if, so it never got the "touched" column
    assert pd.isna(touched_by_text["drop"])


def test_run_if_condition_does_not_prevent_deletions_from_full_mapping():
    class DeleteStage(BaseStage):
        def __call__(self, mapping: RecordMapping) -> None:
            for key in list(mapping.keys()):
                del mapping[key]

    df = pd.DataFrame({"text": ["a", "b"]})
    pipeline = NLPPipeline(stages=[DeleteStage()])
    result = pipeline(df)

    assert len(result.df) == 0


def test_uuid_generator_resets_between_runs():
    from nlp_pipeline.uuid_gen import default_uuid_generator

    df = pd.DataFrame({"text": ["a"]})
    captured_keys = []

    class CaptureKeyStage(BaseStage):
        def __call__(self, mapping: RecordMapping) -> None:
            captured_keys.append(list(mapping.keys())[0])

    pipeline = NLPPipeline(stages=[CaptureKeyStage()])
    pipeline(df)
    pipeline(df)

    # same seeded generator + a reset each run => identical first key both times
    assert captured_keys[0] == captured_keys[1]
    assert default_uuid_generator() is default_uuid_generator()
