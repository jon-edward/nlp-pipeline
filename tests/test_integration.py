import pandas as pd

from nlp_pipeline import (
    KeywordExtractionStage,
    NLPPipeline,
    TextClassificationStage,
    TextClusteringStage,
    TfidfVectorizeStage,
)


def test_full_pipeline_with_all_stock_stages_runs_end_to_end():
    df = pd.DataFrame(
        {
            "text": [
                "I absolutely love this product, amazing quality",
                "This is terrible, worst purchase I ever made",
                "Cats and dogs are wonderful household pets",
                "Stocks and bonds are common investment vehicles",
            ]
        }
    )

    classifier = TextClassificationStage(text_field="text").fit(
        ["amazing wonderful love great", "terrible worst awful bad"],
        ["positive", "negative"],
    )

    pipeline = NLPPipeline(
        stages=[
            TfidfVectorizeStage(text_field="text", output_field="tfidf"),
            KeywordExtractionStage(text_field="text", top_k=3),
            TextClusteringStage(text_field="text", n_clusters=2, random_state=0),
            classifier,
        ]
    )

    result = pipeline(df)

    assert len(result.df) == 4
    for column in ["tfidf", "keywords", "cluster", "predicted_label"]:
        assert column in result.df.columns
        assert result.df[column].notna().all()


def test_pipeline_stages_compose_without_interfering_with_each_other():
    df = pd.DataFrame({"text": ["hello world", "goodbye world"]})

    pipeline = NLPPipeline(
        stages=[
            TfidfVectorizeStage(text_field="text", output_field="vec_a"),
            TfidfVectorizeStage(text_field="text", output_field="vec_b", as_dict=False),
        ]
    )
    result = pipeline(df)

    assert "vec_a" in result.df.columns
    assert "vec_b" in result.df.columns
    assert all(isinstance(v, dict) for v in result.df["vec_a"])
    assert all(isinstance(v, list) for v in result.df["vec_b"])
