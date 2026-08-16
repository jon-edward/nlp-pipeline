"""The abstract base class every pipeline stage derives from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from nlp_pipeline.conditions import Condition, NoopCondition
from nlp_pipeline.records import RecordMapping


@dataclass
class BaseStage(ABC):
    """Base class for conditional pipeline stages.

    Subclasses implement :meth:`__call__` to perform their work.
    """

    run_if: Condition = field(default_factory=NoopCondition, kw_only=True)
    """Determines, per-record, whether :meth:`__call__` should receive the record."""

    @abstractmethod
    def __call__(self, mapping: RecordMapping) -> None:
        """Perform the stage's work on the records in ``mapping``.

        This method should not return anything, and should modify ``mapping`` in-place through standard
        dict operations (e.g. ``__setitem__``, ``__delitem__``). See :class:`RecordMapping` for specialized
        convenience methods.
        """
        raise NotImplementedError
