"""A mapping from UUIDs to records."""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView
from typing import Any, Hashable, TypeAlias, MutableMapping, Iterator
import uuid

DictRecord: TypeAlias = MutableMapping[Hashable, Any]


class _RecordMapping(MutableMapping):
    """
    Shared implementation for anything backed by an underlying dict-like
    object: RecordStore is backed by its own dict, RecordSlice is backed by
    a parent RecordStore/RecordSlice. Subclasses only need to define
    `_backing` (the mapping actually holding the data) and can override the
    `_visible_keys` / `_on_set` / `_on_delete` hooks below.
    """

    _backing: MutableMapping[str, DictRecord]

    def __init__(self, initial: MutableMapping[str, DictRecord] | None = None):
        self._backing = dict(initial) if initial else {}

    def __getitem__(self, key: str) -> DictRecord:
        return self._backing[key]

    def __setitem__(self, key: str, value: dict) -> None:
        self._backing[key] = value
        self._on_set(key)

    def __delitem__(self, key: str) -> None:
        del self._backing[key]
        self._on_delete(key)

    def __contains__(self, key: object) -> bool:
        return key in self._backing

    def __iter__(self) -> Iterator[str]:
        return self._visible_keys()

    def __len__(self) -> int:
        return sum(1 for _ in self._visible_keys())

    # hooks subclasses can override; no-ops
    def _on_set(self, key: str) -> None:
        pass

    def _on_delete(self, key: str) -> None:
        pass

    def _visible_keys(self) -> Iterator[str]:
        return iter(self._backing)

    # shared convenience
    def add(self, record: dict, key: str | None = None) -> str:
        """Add a new record, generating a uuid if not provided. Returns the key."""
        key = key or uuid.uuid4().hex
        self[key] = record
        return key

    def slice(self, keys: set[str] | None = None) -> RecordSlice:
        """Create a slice restricted to a subset of this mapping's keys."""
        return RecordSlice(self, keys or set())

    def items(self) -> ItemsView[str, DictRecord]:
        return ItemsView(self)

    def keys(self) -> KeysView[str]:
        return KeysView(self)

    def values(self) -> ValuesView[DictRecord]:
        return ValuesView(self)

    def __repr__(self) -> str:
        dict_repr = ", ".join(f"{k}: {v!r}" for k, v in self.items())
        return f"{self.__class__.__name__}({{{dict_repr}}})"


class RecordStore(_RecordMapping):
    """A mapping of uuid (str) -> dict, backed by a plain dict."""

    def __init__(self, initial: MutableMapping[str, DictRecord] | None = None):
        self._backing = dict(initial) if initial else {}


class RecordSlice(_RecordMapping):
    """
    A view over a RecordStore (or another RecordSlice) restricted to a
    subset of keys for the purposes of iteration/len/repr — but
    get/set/delete operate directly on the underlying store, so:
      - reading any key in the underlying store works, even if that key
        isn't part of this slice's visible subset (use .untrack() if you
        want to hide a key without deleting it)
      - setting a key (new or existing) adds it to the visible subset,
        matching normal MutableMapping semantics
      - deleting a key removes it from the underlying store AND from the
        slice's visible subset, if present
    """

    _keys: set[str]

    def __init__(self, backing: _RecordMapping | None = None, keys: set[str] | None = None):
        self._backing = backing or {}
        self._keys = keys or set()

    def _visible_keys(self) -> Iterator[str]:
        return iter(k for k in self._keys if k in self._backing)

    def _on_set(self, key: str) -> None:
        self._keys.add(key)

    def _on_delete(self, key: str) -> None:
        self._keys.discard(key)

    def track(self, key: str) -> None:
        """Add an existing (or future) key to this slice's visible subset."""
        self._keys.add(key)

    def untrack(self, key: str) -> None:
        """Remove a key from this slice's visible subset without deleting it."""
        self._keys.discard(key)
