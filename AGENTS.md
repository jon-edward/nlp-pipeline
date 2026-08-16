# AGENTS.md

Guidance for AI coding agents (and humans skimming for the same info) working
in this repository.

## What this repo is

`nlp_pipeline` is a small orchestration layer for running a sequence of
"stages" (QA, classification, clustering, keyword extraction, ...) over a
pandas `DataFrame`, one record at a time. Records are plain dicts keyed by a
generated UUID; stages read and mutate those dicts in place.

Core pieces, in dependency order:

| Module | Purpose |
|---|---|
| `nlp_pipeline/uuid_gen.py` | Seeded, reproducible UUID generation. |
| `nlp_pipeline/records.py` | `RecordMapping` (UUID -> record dict) and `RecordMappingSlice` (a filtered view over one). |
| `nlp_pipeline/conditions.py` | `Condition` subclasses (`ColumnCondition`, `ConditionGroup`, `FunctionCondition`, `NoopCondition`) used by stages' `run_if`. |
| `nlp_pipeline/stages/base.py` | `BaseStage`: the abstract base every stage extends. |
| `nlp_pipeline/stages/*.py` | Concrete stage implementations, including the scikit-learn-backed ones (`vectorize.py`, `classify.py`, `cluster.py`, `keywords.py`). |
| `nlp_pipeline/pipeline.py` | `NLPPipeline`: runs stages over a dataframe in order. |

## Setup

```bash
pip install -e ".[dev]"
```

This installs `nlp_pipeline` in editable mode plus `pytest` / `pytest-cov`.
Requires Python >= 3.10 (the code uses `from __future__ import annotations`
and PEP 604 unions like `int | None`, but is written to remain compatible
with 3.10+).

## Running tests

```bash
pytest                          # unit tests + module doctests (see pyproject.toml)
pytest --cov=nlp_pipeline -q    # with coverage
pytest tests/stages/test_cluster.py -q   # a single file
pytest -k "classification"      # by keyword
```

Doctests inside `nlp_pipeline/**/*.py` are collected automatically
(`--doctest-modules` is set in `pyproject.toml`'s `[tool.pytest.ini_options]`).
If you add or edit a docstring `Examples` block with a `>>>` prompt, it will
run as a test; keep those runnable, or mark unrunnable pseudocode with
`# doctest: +SKIP` (see the example in `pipeline.py`).

## Conventions to follow when extending this codebase

1. **Stages are dataclasses that extend `BaseStage`.** Implement `__call__(self, mapping: RecordMapping) -> None`
   and mutate records in place; don't return a new dataframe or reassign
   `mapping` itself. `run_if` is provided for free by the base class; don't
   re-implement per-record filtering inside your stage.

2. **Guard against empty batches.** `NLPPipeline` narrows the mapping passed to
   a stage down to whatever `run_if` matches, which can be zero records. The
   sklearn-backed stages all early-return on an empty batch (see
   `stages/_sklearn_utils.py`'s `require_non_empty`) instead of letting
   scikit-learn raise on an empty vocabulary. Follow the same pattern in new
   stages.

3. **Classification-style (supervised) stages are fit out-of-band.**
   `TextClassificationStage` is *not* fit as part of a pipeline run, call
   `.fit(texts, labels)` before adding it to `NLPPipeline.stages` and it
   raises `RuntimeError` if used unfit. Unsupervised stages
   (`TfidfVectorizeStage`, `TextClusteringStage`, `KeywordExtractionStage`)
   fit fresh on every call by default since there's no external label set to
   learn ahead of time. If you add a new supervised stage, follow the
   `TextClassificationStage` pattern rather than trying to infer labels from
   the dataframe implicitly.

4. **Field names are constructor parameters, not hardcoded strings.** Every
   stage takes `text_field` (or similar) for input and `output_field` for
   output so multiple instances of the same stage can coexist in one
   pipeline (see `test_pipeline_stages_compose_without_interfering_with_each_other`
   in `tests/test_integration.py`).

5. **Don't use `nlp_pipeline.uuid_gen.default_uuid_generator()` directly in
   new stages.** It's reserved for `NLPPipeline.__call__`'s record-key
   generation and is a shared, seeded, process-global singleton. Stages that
   need to add new records to the mapping should use
   `mapping.add(record)` (see `RecordMapping.add`), which delegates to it
   correctly without stages needing to know about UUID generation at all.

6. **Match existing docstring style.** Every public class/function here has a
   NumPy-style docstring with a short one-line summary, an extended
   description where useful, and often a runnable `Examples` doctest. New
   stages should follow suit, it's also free test coverage.

7. **Imports are absolute, not relative** (`from nlp_pipeline.records import
   ...`, not `from .records import ...` or the bare `from records import ...`
   that this repo originally shipped with and that broke every import outside
   of one specific working directory, see "Known issues" below). Always
   import via the full `nlp_pipeline.` prefix.

## Known issues / things not to be surprised by

- `RecordMappingSlice.__contains__` is inherited from `RecordMapping` and
  checks `key in self._backing`, which for a slice is its *parent* mapping, 
  not the slice's own tracked-keys set. This means `"x" in some_slice` and
  `"x" in list(some_slice)` can legitimately disagree. This is exercised
  (not silently patched) in `tests/test_records.py::TestRecordMappingSlice::test_contains_checks_the_backing_mapping_not_slice_visibility`.
  If you're relying on slice membership semantics, iterate the slice rather
  than using `in`.
- Clustering and keyword extraction re-fit their scikit-learn models on every
  call using whatever batch of records they're given, so cluster IDs and
  keyword rankings are only meaningful *within* a single pipeline run, not
  across separate runs or across differently-filtered batches in the same
  run.

## Before submitting a change

- Run `pytest` and make sure it's green.
- If you touched `records.py`, `conditions.py`, or `pipeline.py`, also run
  `pytest --doctest-modules nlp_pipeline -q` explicitly, since those modules
  carry doctests that are easy to break with a signature change.
- Prefer adding a focused unit test over extending an existing test with more
  assertions; the test files here are organized one-class/behavior-per-test
  intentionally, so failures point at exactly what broke.
