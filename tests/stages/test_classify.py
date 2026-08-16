import pytest

from nlp_pipeline.records import RecordMapping
from nlp_pipeline.stages.classify import TextClassificationStage

TRAIN_TEXTS = [
    "amazing product love it",
    "wonderful experience great job",
    "terrible awful hated it",
    "worst purchase ever bad",
]
TRAIN_LABELS = ["positive", "positive", "negative", "negative"]


def test_raises_if_used_before_fit():
    stage = TextClassificationStage(text_field="text")
    mapping = RecordMapping({"a": {"text": "hello"}})

    with pytest.raises(RuntimeError):
        stage(mapping)


def test_fit_returns_self_for_chaining():
    stage = TextClassificationStage(text_field="text")
    result = stage.fit(TRAIN_TEXTS, TRAIN_LABELS)
    assert result is stage


def test_predicts_a_known_label_for_every_record():
    stage = TextClassificationStage(text_field="text").fit(TRAIN_TEXTS, TRAIN_LABELS)
    mapping = RecordMapping(
        {
            "a": {"text": "amazing wonderful"},
            "b": {"text": "terrible worst"},
        }
    )
    stage(mapping)

    for key in mapping:
        assert mapping[key]["predicted_label"] in {"positive", "negative"}


def test_writes_confidence_between_zero_and_one():
    stage = TextClassificationStage(text_field="text").fit(TRAIN_TEXTS, TRAIN_LABELS)
    mapping = RecordMapping({"a": {"text": "amazing product"}})
    stage(mapping)

    confidence = mapping["a"]["predicted_label_confidence"]
    assert 0.0 <= confidence <= 1.0


def test_output_proba_field_none_skips_confidence():
    stage = TextClassificationStage(text_field="text", output_proba_field=None).fit(
        TRAIN_TEXTS, TRAIN_LABELS
    )
    mapping = RecordMapping({"a": {"text": "amazing product"}})
    stage(mapping)

    assert "predicted_label" in mapping["a"]
    assert "predicted_label_confidence" not in mapping["a"]


def test_custom_output_field_names_are_respected():
    stage = TextClassificationStage(
        text_field="text",
        output_field="sentiment",
        output_proba_field="sentiment_score",
    ).fit(TRAIN_TEXTS, TRAIN_LABELS)
    mapping = RecordMapping({"a": {"text": "amazing product"}})
    stage(mapping)

    assert "sentiment" in mapping["a"]
    assert "sentiment_score" in mapping["a"]


def test_empty_mapping_is_a_noop_after_fit():
    stage = TextClassificationStage(text_field="text").fit(TRAIN_TEXTS, TRAIN_LABELS)
    mapping = RecordMapping({})
    stage(mapping)  # should not raise
    assert len(mapping) == 0
