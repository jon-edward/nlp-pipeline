import pytest

from nlp_pipeline.records import RecordMapping, RecordMappingSlice


@pytest.fixture
def mapping():
    return RecordMapping(
        {
            "a": {"x": 1},
            "b": {"x": 2},
            "c": {"x": 3},
        }
    )


class TestRecordMapping:
    def test_getitem(self, mapping: RecordMapping):
        assert mapping["a"] == {"x": 1}

    def test_setitem_adds_new_key(self, mapping: RecordMapping):
        mapping["d"] = {"x": 4}
        assert mapping["d"] == {"x": 4}
        assert len(mapping) == 4

    def test_delitem_removes_key(self, mapping: RecordMapping):
        del mapping["a"]
        assert "a" not in mapping
        assert len(mapping) == 2

    def test_contains(self, mapping: RecordMapping):
        assert "a" in mapping
        assert "z" not in mapping

    def test_iter_and_len(self, mapping: RecordMapping):
        assert set(iter(mapping)) == {"a", "b", "c"}
        assert len(mapping) == 3

    def test_add_generates_a_key(self, mapping: RecordMapping):
        key = mapping.add({"x": 99})
        assert isinstance(key, str)
        assert mapping[key] == {"x": 99}

    def test_add_with_explicit_key(self, mapping: RecordMapping):
        key = mapping.add({"x": 99}, key="explicit")
        assert key == "explicit"
        assert mapping["explicit"] == {"x": 99}

    def test_repr_is_sorted_and_readable(self, mapping: RecordMapping):
        text = repr(mapping)
        assert text.startswith("RecordMapping({")
        assert "'a': {'x': 1}" in text

    def test_to_dict_on_plain_dict_backing_returns_the_backing_dict(
        self, mapping: RecordMapping
    ):
        # per the implementation, backed-by-dict mappings return the live
        # dict rather than a copy
        assert mapping.to_dict() is mapping.backing

    def test_slice_to_dict_filters_to_only_visible_keys(self, mapping: RecordMapping):
        sl = mapping.slice({"a", "b"})
        assert sl.to_dict() == {"a": {"x": 1}, "b": {"x": 2}}

    def test_mutablemapping_mixins_work(self, mapping: RecordMapping):
        # get/keys/values/items/update come from MutableMapping for free
        assert mapping.get("a") == {"x": 1}
        assert mapping.get("missing", "default") == "default"
        assert set(mapping.keys()) == {"a", "b", "c"}
        mapping.update({"e": {"x": 5}})
        assert mapping["e"] == {"x": 5}


class TestRecordMappingSlice:
    def test_slice_only_exposes_given_keys(self, mapping: RecordMapping):
        sl = mapping.slice({"a", "b"})
        assert set(iter(sl)) == {"a", "b"}
        assert len(sl) == 2

    def test_slice_get_reads_through_to_backing(self, mapping: RecordMapping):
        sl = mapping.slice({"a"})
        assert sl["a"] == {"x": 1}

    def test_slice_set_on_existing_key_writes_through(self, mapping: RecordMapping):
        sl = mapping.slice({"a"})
        sl["a"] = {"x": 100}
        assert mapping["a"] == {"x": 100}

    def test_slice_set_on_new_key_adds_to_backing_and_tracks_it(
        self, mapping: RecordMapping
    ):
        sl = mapping.slice({"a"})
        sl["z"] = {"x": 999}

        assert mapping["z"] == {"x": 999}
        assert "z" in sl

    def test_slice_delete_removes_from_backing_and_untracks(
        self, mapping: RecordMapping
    ):
        sl = mapping.slice({"a", "b"})
        del sl["a"]

        assert "a" not in mapping
        assert "a" not in sl

    def test_track_adds_key_to_iteration(self, mapping: RecordMapping):
        sl = mapping.slice(set())
        assert "a" not in list(iter(sl))

        sl.track("a")
        assert "a" in list(iter(sl))
        assert sl["a"] == {"x": 1}

    def test_untrack_hides_from_iteration_without_deleting(
        self, mapping: RecordMapping
    ):
        sl = mapping.slice({"a", "b"})
        sl.untrack("a")

        assert "a" not in list(iter(sl))
        assert "a" in mapping  # still present in the backing mapping

    def test_contains_checks_the_backing_mapping_not_slice_visibility(
        self, mapping: RecordMapping
    ):
        # NB: __contains__ is inherited from RecordMapping and checks
        # `key in self._backing`, which for a slice is its parent mapping --
        # not the slice's own tracked-keys set. This means `in` and `iter()`
        # can disagree for a slice (see the improvements notes in README.md).
        sl = mapping.slice(set())
        assert "a" not in list(iter(sl))
        assert "a" in sl  # backing (the full `mapping`) does contain "a"

    def test_slice_of_slice_drills_through_to_root_backing(
        self, mapping: RecordMapping
    ):
        sl1 = mapping.slice({"a"})
        sl2 = sl1.slice(set())

        sl2["d"] = {"x": 4}

        # setting through sl2 should reach sl1's backing, and therefore mapping
        assert mapping["d"] == {"x": 4}

    def test_keys_not_in_backing_are_invisible(self, mapping: RecordMapping):
        sl = mapping.slice({"a", "ghost"})
        assert set(iter(sl)) == {"a"}
        assert len(sl) == 1
