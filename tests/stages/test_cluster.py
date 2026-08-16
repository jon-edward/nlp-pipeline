from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.cluster import TextClusteringStage


def _mapping():
    return RecordMapping(
        {
            "a": {"text": "cats and dogs are pets"},
            "b": {"text": "dogs and cats are pets too"},
            "c": {"text": "stocks and bonds are investments"},
            "d": {"text": "bonds and stocks are financial investments"},
        }
    )


def test_writes_a_cluster_id_to_every_record():
    stage = TextClusteringStage(text_field="text", n_clusters=2, random_state=0)
    mapping = _mapping()
    stage(mapping)

    for key in mapping:
        assert isinstance(mapping[key]["cluster"], int)


def test_similar_texts_land_in_the_same_cluster():
    stage = TextClusteringStage(text_field="text", n_clusters=2, random_state=0)
    mapping = _mapping()
    stage(mapping)

    assert mapping["a"]["cluster"] == mapping["b"]["cluster"]
    assert mapping["c"]["cluster"] == mapping["d"]["cluster"]
    assert mapping["a"]["cluster"] != mapping["c"]["cluster"]


def test_n_clusters_is_capped_at_number_of_records():
    stage = TextClusteringStage(text_field="text", n_clusters=10, random_state=0)
    mapping = RecordMapping({"a": {"text": "only one record here"}})
    stage(mapping)  # would raise from sklearn if not capped

    assert mapping["a"]["cluster"] == 0


def test_empty_mapping_is_a_noop_and_does_not_raise():
    stage = TextClusteringStage(text_field="text")
    mapping = RecordMapping({})
    stage(mapping)
    assert len(mapping) == 0


def test_custom_output_field_is_respected():
    stage = TextClusteringStage(
        text_field="text", output_field="topic", n_clusters=2, random_state=0
    )
    mapping = _mapping()
    stage(mapping)

    assert "topic" in mapping["a"]
    assert "cluster" not in mapping["a"]
