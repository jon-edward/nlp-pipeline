"""Internal helpers shared by the scikit-learn-backed stages in this package."""

from __future__ import annotations

from typing import Sequence

from nlp_pipeline.records import RecordMapping


def collect_texts(
    mapping: RecordMapping, text_field: str, default: str = ""
) -> tuple[list[str], list[str]]:
    """Pull ``(keys, texts)`` out of ``mapping`` for the given field.

    Missing or ``None`` values are coerced to ``default`` so a single record
    with a hole in it doesn't blow up an entire batch fit/transform call.
    """
    keys = list(mapping.keys())
    texts = [str(mapping[key].get(text_field, default) or default) for key in keys]
    return keys, texts
