from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.vectorize import TfidfVectorizeStage


def _mapping():
    return RecordMapping(
        {
            "a": {"text": "cats and dogs are pets"},
            "b": {"text": "dogs and cats are pets"},
            "c": {"text": "stocks and bonds are investments"},
        }
    )


def test_writes_a_vector_field_to_every_record():
    stage = TfidfVectorizeStage(text_field="text", output_field="vec")
    mapping = _mapping()
    stage(mapping)

    for key in mapping:
        assert "vec" in mapping[key]


def test_as_dict_output_contains_only_nonzero_terms_present_in_text():
    stage = TfidfVectorizeStage(text_field="text", output_field="vec", as_dict=True)
    mapping = _mapping()
    stage(mapping)

    vec_a = mapping["a"]["vec"]
    assert isinstance(vec_a, dict)
    assert "cats" in vec_a
    assert "stocks" not in vec_a  # never appears in record "a"'s text


def test_as_dict_false_returns_dense_vector_of_consistent_length():
    stage = TfidfVectorizeStage(text_field="text", output_field="vec", as_dict=False)
    mapping = _mapping()
    stage(mapping)

    lengths = {len(mapping[key]["vec"]) for key in mapping}
    assert len(lengths) == 1  # every record's vector is the same length


def test_similar_texts_are_closer_than_dissimilar_ones():
    stage = TfidfVectorizeStage(text_field="text", output_field="vec", as_dict=False)
    mapping = _mapping()
    stage(mapping)

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v))

    sim_ab = dot(mapping["a"]["vec"], mapping["b"]["vec"])
    sim_ac = dot(mapping["a"]["vec"], mapping["c"]["vec"])
    assert sim_ab > sim_ac


def test_empty_mapping_is_a_noop_and_does_not_raise():
    stage = TfidfVectorizeStage(text_field="text")
    mapping = RecordMapping({})
    stage(mapping)  # should not raise on an empty batch
    assert len(mapping) == 0


def test_missing_text_field_defaults_to_empty_string():
    stage = TfidfVectorizeStage(text_field="text", output_field="vec")
    mapping = RecordMapping({"a": {}, "b": {"text": "hello world"}})
    stage(mapping)

    assert mapping["a"]["vec"] == {}


def test_refit_false_uses_pre_fit_vectorizer_vocabulary_only():
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(["cats and dogs are pets"])

    stage = TfidfVectorizeStage(
        text_field="text", vectorizer=vectorizer, refit=False, as_dict=True
    )
    mapping = RecordMapping({"a": {"text": "stocks and bonds are investments"}})
    stage(mapping)

    # none of these words were in the pre-fit vocabulary, so the vector is empty
    assert mapping["a"]["tfidf_vector"] == {}
