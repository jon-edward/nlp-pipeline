# nlp_pipeline

A small orchestration layer for composing NLP stages (embedding,
classification, clustering, keyword extraction, ...) over a pandas
`DataFrame`. You define an ordered list of stages, each stage reads and
writes fields on a per-record basis, and the pipeline hands you back a
dataframe with everything merged in.

```python
import pandas as pd
from nlp_pipeline import NLPPipeline, TfidfVectorizeStage, TextClusteringStage

df = pd.DataFrame(
    {
        "text": [
            "I absolutely love this product",
            "This is the worst purchase I ever made",
            "Cats and dogs are wonderful pets",
        ]
    }
)

pipeline = NLPPipeline(
    stages=[
        TfidfVectorizeStage(text_field="text", output_field="tfidf"),
        TextClusteringStage(text_field="text", n_clusters=2, random_state=0),
    ]
)

result = pipeline(df)
print(result.df[["text", "cluster"]])
```

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+. Core runtime dependencies are `pandas` and
`scikit-learn`; `pytest`/`pytest-cov` are dev-only (see `pyproject.toml`).

## Concepts

- **`NLPPipeline`** (`pipeline.py`) — runs a list of stages, in order, over a
  dataframe. Internally it converts each row into a UUID-keyed record dict
  (via a reproducible, seeded `UUIDGenerator`) and reassembles a dataframe
  from those records once every stage has run.
- **`BaseStage`** (`stages/base.py`) — the abstract base every stage extends.
  A stage implements `__call__(self, mapping) -> None` and mutates records in
  place. Every stage also accepts a `run_if: Condition` that filters which
  records it sees, so you can e.g. only classify English-language rows.
- **`Condition`** (`conditions.py`) — small composable predicates
  (`ColumnCondition`, `ConditionGroup`, `FunctionCondition`, `NoopCondition`)
  used for the `run_if` gating above.
- **`RecordMapping` / `RecordMappingSlice`** (`records.py`) — a dict-like
  UUID -> record mapping, plus a filtered "view" over a subset of keys that
  writes through to the full mapping. This is what `run_if` filtering
  produces under the hood.

## Built-in scikit-learn stages (`stages/`)

| Stage | What it does | Fit timing |
|---|---|---|
| `TfidfVectorizeStage` | Embeds each record's text as a TF-IDF vector (as a `{term: weight}` dict, or a dense `list[float]`). | Refits per call by default; pass a pre-fit `vectorizer` + `refit=False` to reuse a fixed vocabulary. |
| `TextClassificationStage` | Predicts a label (+ confidence) per record using a scikit-learn classifier (`TfidfVectorizer` + `LogisticRegression` by default). | Must be fit ahead of time via `.fit(texts, labels)` — raises `RuntimeError` if used unfit. |
| `TextClusteringStage` | Groups records into `n_clusters` groups via TF-IDF + `KMeans`. | Fits fresh on every call's batch (cluster IDs are only comparable within one run). |
| `KeywordExtractionStage` | Extracts each record's top-`k` TF-IDF-weighted terms, relative to the other records in the same call. | Fits fresh on every call's batch. |

All four:
- take `text_field` (where to read text from) and `output_field` (where to
  write results), so you can run the same stage type multiple times with
  different field names in one pipeline.
- no-op safely on an empty batch (e.g. if a `run_if` condition filters
  everything out) instead of raising scikit-learn's empty-vocabulary error.
- have runnable doctests in their docstrings — see the source for quick
  usage examples.

### Classification example

```python
from nlp_pipeline import TextClassificationStage

clf = TextClassificationStage(text_field="text").fit(
    texts=["amazing product", "terrible service"],
    labels=["positive", "negative"],
)
# now add `clf` to an NLPPipeline's stages list
```

### Gating a stage with `run_if`

```python
from nlp_pipeline import ColumnCondition, ColumnConditionOp, TextClusteringStage

stage = TextClusteringStage(
    text_field="text",
    n_clusters=3,
    run_if=ColumnCondition("language", ColumnConditionOp.EQ, "en"),
)
```

## Testing

```bash
pytest                        # full suite (unit tests + doctests), 92 passed
pytest --cov=nlp_pipeline -q  # with coverage (currently ~96%)
```

See `AGENTS.md` for conventions to follow when adding new stages or tests.
