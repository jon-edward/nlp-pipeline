from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.keywords import KeywordExtractionStage


def test_writes_a_keyword_list_to_every_record():
    stage = KeywordExtractionStage(text_field="text", top_k=3)
    mapping = RecordMapping(
        {
            "a": {"text": "the quick brown fox jumps over the lazy dog"},
            "b": {"text": "the lazy dog sleeps all day long"},
        }
    )
    stage(mapping)

    for key in mapping:
        assert isinstance(mapping[key]["keywords"], list)
        assert len(mapping[key]["keywords"]) <= 3


def test_distinctive_terms_are_favored_over_shared_terms():
    stage = KeywordExtractionStage(text_field="text", top_k=1)
    # equal term frequency for "shared" and the unique term in each doc, so the
    # comparison isolates IDF: "shared" appears in every document (idf lower)
    # while each unique term appears in only one (idf higher), so it should win
    mapping = RecordMapping(
        {
            "a": {"text": "shared unique_a"},
            "b": {"text": "shared unique_b"},
        }
    )
    stage(mapping)

    assert mapping["a"]["keywords"] == ["unique_a"]
    assert mapping["b"]["keywords"] == ["unique_b"]


def test_top_k_limits_number_of_keywords():
    stage = KeywordExtractionStage(text_field="text", top_k=2)
    mapping = RecordMapping({"a": {"text": "alpha beta gamma delta epsilon"}})
    stage(mapping)

    assert len(mapping["a"]["keywords"]) == 2


def test_empty_mapping_is_a_noop_and_does_not_raise():
    stage = KeywordExtractionStage(text_field="text")
    mapping = RecordMapping({})
    stage(mapping)
    assert len(mapping) == 0


def test_custom_output_field_is_respected():
    stage = KeywordExtractionStage(text_field="text", output_field="top_terms")
    mapping = RecordMapping({"a": {"text": "alpha beta gamma"}})
    stage(mapping)

    assert "top_terms" in mapping["a"]
    assert "keywords" not in mapping["a"]
