"""A mapping from UUIDs to records."""

from __future__ import annotations

from typing import Any, Hashable, TypeAlias, MutableMapping, Iterator, Iterable

from nlp_pipeline.uuid_gen import default_uuid_generator

DictRecord: TypeAlias = MutableMapping[Hashable, Any]


class RecordMapping(MutableMapping[str, DictRecord]):
    """
    A mapping from UUIDs to records.

    This is backed by a plain dict, but subclasses can override the underlying
    ``_on_set`` / ``_on_delete`` hooks to provide additional behavior.
    """

    _backing: MutableMapping[str, DictRecord]

    def __init__(self, initial: dict[str, DictRecord]):
        self._backing = initial

    @property
    def backing(self) -> MutableMapping[str, DictRecord]:
        return self._backing

    def __getitem__(self, key: str) -> DictRecord:
        return self._backing[key]

    def __setitem__(self, key: str, value: DictRecord) -> None:
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

    def slice(self, keys: set[str]) -> RecordMappingSlice:
        """Create a slice restricted to a subset of this mapping's keys."""
        return RecordMappingSlice(self, keys)

    def add(self, value: DictRecord, key: str | None = None) -> str:
        if key is None:
            key = default_uuid_generator().next()
        self[key] = value
        return key

    def __repr__(self) -> str:
        dict_repr = ", ".join(f"{k!r}: {self[k]!r}" for k in sorted(self))
        return f"{self.__class__.__name__}({{{dict_repr}}})"

    def to_dict(self) -> dict[str, DictRecord]:
        if isinstance(self._backing, dict):
            # no need to copy
            return self._backing
        # filter out to only visible keys
        return {k: v for k, v in self.items()}


class RecordMappingSlice(RecordMapping):
    """
    A view over a RecordMapping (or another RecordSlice) restricted to a
    subset of keys for the purposes of iteration/len/repr but
    get/set/delete operate directly on the underlying mapping.

    Do not instantiate directly, use :meth:`RecordMapping.slice` instead.
    """

    _keys: set[str]

    def __init__(self, backing: RecordMapping, keys: Iterable[str]):
        self._backing = backing
        self._keys = set(keys)

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


if __name__ == "__main__":
    mapping = RecordMapping(
        {
            "a": {"a": 1, "b": 2},
            "b": {"a": 3, "b": 4},
        }
    )

    mapping_slice_1 = mapping.slice({"a"})
    mapping_slice_1.update({"c": {"a": 5, "b": 6}})

    mapping_slice_2 = mapping_slice_1.slice(set())
    mapping_slice_2["d"] = {"a": 7, "b": 8}
    # note that setting "d" drills down to mapping_slice_1 and mapping because
    # it can be reached via backing

    print("=== All keys ===")
    print(mapping, "\n")

    print("=== Slice 1 ('a', 'd') ===")
    print(mapping_slice_1, "\n")

    print("=== Slice 2 ('d') ===")
    print(mapping_slice_2, "\n")
