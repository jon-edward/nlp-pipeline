import pytest

from nlp_pipeline.conditions import (
    ColumnCondition,
    ColumnConditionOp,
    ConditionGroup,
    ConditionGroupOp,
    FunctionCondition,
    NoopCondition,
)


class TestColumnCondition:
    @pytest.mark.parametrize(
        "op,value,record_value,expected",
        [
            (ColumnConditionOp.EQ, "en", "en", True),
            (ColumnConditionOp.EQ, "en", "fr", False),
            (ColumnConditionOp.NE, "en", "fr", True),
            (ColumnConditionOp.GT, 10, 20, True),
            (ColumnConditionOp.GT, 10, 5, False),
            (ColumnConditionOp.LT, 10, 5, True),
            (ColumnConditionOp.GE, 10, 10, True),
            (ColumnConditionOp.LE, 10, 10, True),
            (ColumnConditionOp.CONTAINS, "cat", "the cats sat", True),
            (ColumnConditionOp.CONTAINS, "dog", "the cats sat", False),
        ],
    )
    def test_ops(self, op, value, record_value, expected):
        cond = ColumnCondition("field", op, value)
        assert cond.matches({"field": record_value}) is expected

    def test_is_op_uses_identity(self):
        cond = ColumnCondition("field", ColumnConditionOp.IS, None)
        assert cond.matches({"field": None}) is True
        assert cond.matches({"field": 0}) is False

    def test_missing_column_defaults_to_none_comparison(self):
        cond = ColumnCondition("missing", ColumnConditionOp.IS, None)
        assert cond.matches({"other": 1}) is True

    def test_missing_column_raises_when_configured(self):
        cond = ColumnCondition(
            "missing", ColumnConditionOp.EQ, "x", missing_col_key_error=True
        )
        with pytest.raises(KeyError):
            cond.matches({"other": 1})

    def test_type_error_defaults_to_false(self):
        cond = ColumnCondition("field", ColumnConditionOp.GT, "not-a-number")
        assert cond.matches({"field": 5}) is False

    def test_type_error_raises_when_configured(self):
        cond = ColumnCondition(
            "field", ColumnConditionOp.GT, "not-a-number", type_error_as_false=False
        )
        with pytest.raises(TypeError):
            cond.matches({"field": 5})

    def test_unknown_op_raises_value_error(self):
        cond = ColumnCondition("field", "not-a-real-op", "x")
        with pytest.raises(ValueError):
            cond.matches({"field": "x"})


class TestFunctionCondition:
    def test_delegates_to_function(self):
        cond = FunctionCondition(func=lambda record: record["x"] > 0)
        assert cond.matches({"x": 1}) is True
        assert cond.matches({"x": -1}) is False


class TestNoopCondition:
    def test_always_matches(self):
        cond = NoopCondition()
        assert cond.matches({}) is True
        assert cond.matches({"anything": "goes"}) is True


class TestConditionGroup:
    def test_all_requires_every_condition(self):
        group = ConditionGroup(
            conditions=[
                ColumnCondition("a", ColumnConditionOp.EQ, 1),
                ColumnCondition("b", ColumnConditionOp.EQ, 2),
            ],
            op=ConditionGroupOp.ALL,
        )
        assert group.matches({"a": 1, "b": 2}) is True
        assert group.matches({"a": 1, "b": 99}) is False

    def test_any_requires_one_condition(self):
        group = ConditionGroup(
            conditions=[
                ColumnCondition("a", ColumnConditionOp.EQ, 1),
                ColumnCondition("b", ColumnConditionOp.EQ, 2),
            ],
            op=ConditionGroupOp.ANY,
        )
        assert group.matches({"a": 1, "b": 99}) is True
        assert group.matches({"a": 99, "b": 99}) is False

    def test_defaults_to_all(self):
        group = ConditionGroup(conditions=[NoopCondition(), NoopCondition()])
        assert group.op == ConditionGroupOp.ALL

    def test_empty_conditions_all_is_vacuously_true(self):
        group = ConditionGroup(conditions=[], op=ConditionGroupOp.ALL)
        assert group.matches({}) is True

    def test_empty_conditions_any_is_vacuously_false(self):
        group = ConditionGroup(conditions=[], op=ConditionGroupOp.ANY)
        assert group.matches({}) is False

    def test_nested_groups(self):
        inner = ConditionGroup(
            conditions=[
                ColumnCondition("a", ColumnConditionOp.EQ, 1),
                ColumnCondition("b", ColumnConditionOp.EQ, 2),
            ],
            op=ConditionGroupOp.ANY,
        )
        outer = ConditionGroup(
            conditions=[inner, ColumnCondition("c", ColumnConditionOp.EQ, 3)],
            op=ConditionGroupOp.ALL,
        )
        assert outer.matches({"a": 1, "b": 99, "c": 3}) is True
        assert outer.matches({"a": 99, "b": 99, "c": 3}) is False
