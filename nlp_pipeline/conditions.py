"""
Conditions that gate whether a pipeline stage receives a record.

Example
-------
>>> cond = ColumnCondition("language", ColumnConditionOp.EQ, "en")
>>> cond.matches({"language": "en", "text": "hello"})
True
>>> cond.matches({"language": "fr", "text": "bonjour"})
False

Conditions can be combined with :class:`ConditionGroup`:

>>> group = ConditionGroup(
...     conditions=[
...         ColumnCondition("language", ColumnConditionOp.EQ, "en"),
...         ColumnCondition("length", ColumnConditionOp.GT, 10),
...     ],
...     op=ConditionGroupOp.ALL,
... )
>>> group.matches({"language": "en", "length": 20})
True
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence, Literal, final

from nlp_pipeline.records import DictRecord


@dataclass
class Condition(ABC):
    """Abstract base class for all pipeline stage run conditions."""

    @abstractmethod
    def matches(self, record: DictRecord) -> bool:
        """Return whether a record should be passed to an owning stage."""
        raise NotImplementedError


class ColumnConditionOp(str, Enum):
    """A comparison operator used by :class:`ColumnCondition`."""

    GT = "gt"
    LT = "lt"
    NE = "ne"
    EQ = "eq"
    GE = "ge"
    LE = "le"
    CONTAINS = "contains"
    IS = "is"


_TO_FUNC_OP: dict[ColumnConditionOp, Callable[[Any, Any], bool]] = {
    ColumnConditionOp.GT: operator.gt,
    ColumnConditionOp.LT: operator.lt,
    ColumnConditionOp.NE: operator.ne,
    ColumnConditionOp.EQ: operator.eq,
    ColumnConditionOp.GE: operator.ge,
    ColumnConditionOp.LE: operator.le,
    ColumnConditionOp.CONTAINS: operator.contains,
    ColumnConditionOp.IS: operator.is_,
}


@dataclass
class ColumnCondition(Condition):
    """Compares a single record column against a fixed value.

    By default, this will treat missing columns as ``None`` and treat type errors within the comparison as ``False``.
    See :attr:`missing_col_key_error` and :attr:`type_error_as_false` to change this behavior.
    """

    column_name: str
    """The dict key (dataframe column) to read from the record."""

    op: ColumnConditionOp
    """The comparison operator to apply."""

    value: Any
    """The right-hand operand to compare the column's value against."""

    missing_col_key_error: bool = False
    """Whether to raise a KeyError if the column is not present in the record or treat it as ``None``."""

    type_error_as_false: bool = True
    """Whether to treat type errors as ``False``."""

    def matches(self, record: DictRecord) -> bool:
        try:
            call_op = _TO_FUNC_OP[self.op]
        except KeyError:
            raise ValueError(f"Unknown ColumnConditionOp: {self.op}")

        try:
            try:
                return call_op(record[self.column_name], self.value)
            except KeyError:
                if self.missing_col_key_error:
                    raise
                return call_op(None, self.value)
        except TypeError:
            if self.type_error_as_false:
                return False
            raise


@dataclass
class FunctionCondition(Condition):
    """Delegates to a user-supplied function for full custom logic."""

    func: Callable[[DictRecord], bool]
    """The function to delegate to that accepts the full record dict and returns whether it matches."""

    def matches(self, record: DictRecord) -> bool:
        return self.func(record)


@final
@dataclass
class NoopCondition(Condition):
    """A condition that always matches. Used as the default for stages that
    should always run in full."""

    def matches(self, record: DictRecord) -> Literal[True]:
        return True


class ConditionGroupOp(str, Enum):
    """Whether a :class:`ConditionGroup` requires all or any sub-conditions to match."""

    ALL = "all"
    ANY = "any"


@dataclass
class ConditionGroup(Condition):
    """Combines several conditions with ``AND`` (:class:`~ConditionGroupOp.ALL`) or ``OR`` (:class:`~ConditionGroupOp.ANY`) logic.

    Conditions are lazily evaluated in the order they appear in the list.
    """

    conditions: Sequence[Condition]
    """The sub-conditions to evaluate."""

    op: ConditionGroupOp = ConditionGroupOp.ALL
    """Whether every condition must match (``ALL``) or at least one (``ANY``)."""

    def matches(self, record: DictRecord) -> bool:
        if self.op == ConditionGroupOp.ALL:
            return all(condition.matches(record) for condition in self.conditions)
        return any(condition.matches(record) for condition in self.conditions)
