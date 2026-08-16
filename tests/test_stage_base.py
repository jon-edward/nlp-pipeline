import pytest

from nlp_pipeline.conditions import ColumnCondition, ColumnConditionOp, NoopCondition
from nlp_pipeline.stages.base import BaseStage


def test_base_stage_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseStage()  # type: ignore


def test_base_stage_defaults_run_if_to_noop_condition():
    class ConcreteStage(BaseStage):
        def __call__(self, mapping):
            pass

    stage = ConcreteStage()
    assert isinstance(stage.run_if, NoopCondition)


def test_base_stage_accepts_a_custom_run_if_condition():
    class ConcreteStage(BaseStage):
        def __call__(self, mapping):
            pass

    cond = ColumnCondition("field", ColumnConditionOp.EQ, "x")
    stage = ConcreteStage(run_if=cond)
    assert stage.run_if is cond
