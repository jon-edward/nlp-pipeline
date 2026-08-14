# nlp_pipeline

A small orchestration layer for chaining NLP steps (question-answering,
classification, clustering) over a pandas `DataFrame`.

## Layout

```
nlp_pipeline/
  conditions.py             Condition / ColumnCondition / ConditionGroup / etc.
  records.py                RunRecord
  embeddings.py             Embedder / Embedder protocol, SentenceEmbedder, EmbeddingCache (sqlite-vec)
  pipeline.py               NLPPipeline, PipelineRunResult
  stages/
    base.py                 BaseStage
    logistic_classifier.py  LogisticClassifier
tests/
```

## Quick example

```python
from nlp_pipeline.conditions import FunctionCondition
from nlp_pipeline.stages.logistic_classifier import LogisticClassifier
from nlp_pipeline import NLPPipeline

df = ...  # your DataFrame

pipeline = NLPPipeline(
    stages=[
        LogisticClassifier(
            input_column="review",
            label_column="sentiment",
            output_column="model-sentiment",
            run_if=FunctionCondition(lambda r: "type" in r["review"]),
        ),
    ]
)

df = pipeline(df)
print(df.head())
```
