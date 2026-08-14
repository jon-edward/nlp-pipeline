"""The abstract base class every pipeline stage derives from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, final

from tqdm import tqdm

from nlp_pipeline.conditions import Condition, NoopCondition
from nlp_pipeline.constants import ID_COLUMN
from records import RunRecord, DictRecord


@dataclass
class BaseStage(ABC):
    """Base class for all pipeline stages.

    Subclasses implement :meth:`_run_impl` to process a single record.
    :meth:`fit`, if overridden, is called once per stage before any records
    are processed, letting stateful stages (e.g. classifiers) train on the
    full batch first.
    """

    run_if: Condition = field(default_factory=NoopCondition, kw_only=True)
    """
    Determines, per-record, whether :meth:`_run_impl` should perform its
    real work or fall back to default values. Defaults to always-run.
    """

    description: str | None = field(default=None, kw_only=True)
    """
    Human-readable description of this stage, e.g. "Tokenize". Used for progress and logging.
    """

    batch_size: int | None = field(default=None, kw_only=True)
    """
    Number of records to process at once.
    """

    id_column: str = field(init=False, default=ID_COLUMN)
    """
    Internal column name used to track record identity.
    """

    def fit(self, run_records: Sequence[RunRecord]) -> None:
        """Optionally train/prepare this stage using the full batch of :class:`RunRecord` objects.

        Called once by :meth:`nlp_pipeline.pipeline.NLPPipeline.run` before
        any calls to :meth:`run`. The default implementation is a no-op;
        stateful stages (classifiers, cluster models) should override this.

        Parameters
        ----------
        run_records:
            The complete batch of records this stage is about to process.
        """
        return None

    @abstractmethod
    def process_records(self, run_records: Sequence[RunRecord]) -> Sequence[DictRecord]:
        """Process a batch of :class:`RunRecord` and return (possibly augmented) data as zero
        or more :class:`DictRecord` in a sequence. These are then extended into
        run results by :meth:`BaseStage.__call__`.

        Parameters
        ----------
        run_records:
            The records to process, each with a ``do_run`` value whether this stage should
            treat it as eligible for real work.

        Returns
        -------
        Sequence[:class:`DictRecord`]
            Zero or more augmented records.
        """
        raise NotImplementedError

    @final
    def __call__(self, dict_records: Sequence[DictRecord]) -> Sequence[DictRecord]:
        """Apply :meth:`BaseStage.process_records` to every batch of records. Do not override.

        Each record's eligibility is determined by evaluating ``run_if``
        against that record's data (see :class:`RunRecord.do_run`).

        Note that records, regardless of whether they are eligible, are
        passed to :meth:`BaseStage.process_records` and :meth:`BaseStage.fit`. It's
        up to the implementation to determine what eligibility means to each stage.

        Parameters
        ----------
        dict_records:
            The dict records to process, as produced by the previous
            stage (or the initial dataframe, for the first stage).

        Returns
        -------
        Sequence[:class:`DictRecord`]
            The augmented records.
        """

        run_records = [
            RunRecord(record_data=dict_record, do_run=self.run_if.matches(dict_record))
            for dict_record in dict_records
        ]

        self.fit(run_records)

        output = []

        batch_size = self.batch_size or len(run_records)

        for record_idx in range(0, len(run_records), batch_size):
            processed = self.process_records(
                run_records[record_idx : record_idx + batch_size]
            )
            if processed is None:
                continue
            if isinstance(processed, dict):
                output.append(processed)
                continue
            output.extend(processed)

        return output
