"""
Conditions that gate whether a pipeline stage performs real work on a given
record, or falls back to a default value.

Every :class:`Condition` implements a single method, :meth:`Condition.matches`,
which inspects a record (a plain ``dict``) and returns ``True`` if the owning
stage should run its full logic for that record, or ``False`` if the stage
should skip its work and fill in a default instead.

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
from typing import Any, Callable, Sequence

from records import DictRecord


@dataclass
class Condition(ABC):
    """Abstract base class for all pipeline stage run-conditions."""

    @abstractmethod
    def matches(self, record: DictRecord) -> bool:
        """Return whether a stage should perform its full run for ``record``.

        Parameters
        ----------
        record:
            A single row of the pipeline's dataframe, represented as a dict
            mapping column name to value.

        Returns
        -------
        bool
            ``True`` if the stage should do its normal work for this record.
            ``False`` if the stage should skip its work and fill in a
            default value instead (see :meth:`nlp_pipeline.stages.base.BaseStage.process_records`).
        """
        raise NotImplementedError


class ColumnConditionOp(str, Enum):
    """A comparison operator used by :class:`ColumnCondition`."""

    GT = "gt"
    LT = "lt"
    NE = "ne"
    EQ = "eq"
    GE = "ge"
    LE = "le"


_TO_FUNC_OP: dict[ColumnConditionOp, Callable[[Any, Any], bool]] = {
    ColumnConditionOp.GT: operator.gt,
    ColumnConditionOp.LT: operator.lt,
    ColumnConditionOp.NE: operator.ne,
    ColumnConditionOp.EQ: operator.eq,
    ColumnConditionOp.GE: operator.ge,
    ColumnConditionOp.LE: operator.le,
}


@dataclass
class ColumnCondition(Condition):
    """Compares a single record column against a fixed value.

    Attributes
    ----------
    column_name:
        The dict key (dataframe column) to read from the record.
    op:
        The comparison operator to apply.
    value:
        The right-hand operand to compare the column's value against.

    Raises
    ------
    KeyError
        If ``column_name`` is not present in the record.
    """

    column_name: str
    op: ColumnConditionOp
    value: Any

    def matches(self, record: DictRecord) -> bool:
        if self.column_name not in record:
            raise KeyError(
                f"ColumnCondition references missing column '{self.column_name}'. "
                f"Available columns: {', '.join(repr(key) for key in record.keys())}"
            )
        return _TO_FUNC_OP[self.op](record[self.column_name], self.value)


@dataclass
class FunctionCondition(Condition):
    """Delegates to a user-supplied function for full custom logic.

    Attributes
    ----------
    func:
        A callable that accepts the full record dict and returns a bool.
    """

    func: Callable[[DictRecord], bool]

    def matches(self, record: DictRecord) -> bool:
        return self.func(record)


@dataclass
class NoopCondition(Condition):
    """A condition that always matches. Used as the default for stages that
    should always run in full."""

    def matches(self, record: DictRecord) -> bool:
        return True


class ConditionGroupOp(str, Enum):
    """Whether a :class:`ConditionGroup` requires all or any sub-conditions
    to match."""

    ALL = "all"
    ANY = "any"


@dataclass
class ConditionGroup(Condition):
    """Combines several conditions with AND (``ALL``) or OR (``ANY``) logic.

    Attributes
    ----------
    conditions:
        The sub-conditions to evaluate. An empty sequence returns ``True``
        for ``ALL`` (vacuous truth) and ``False`` for ``ANY``.
    op:
        Whether every condition must match (``ALL``) or at least one
        (``ANY``).
    """

    conditions: Sequence[Condition]
    op: ConditionGroupOp = ConditionGroupOp.ALL

    def matches(self, record: DictRecord) -> bool:
        if self.op == ConditionGroupOp.ALL:
            return all(condition.matches(record) for condition in self.conditions)
        return any(condition.matches(record) for condition in self.conditions)
