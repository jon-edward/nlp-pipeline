# Examples

Three larger, runnable pipelines, each built around a public dataset.

| Script | Task | Stages combined | Dataset | Size |
|---|---|---|---|---|
| `01_newsgroups_topic_pipeline.py` | Topic classification + clustering + keywords | `TextClassificationStage`, `TextClusteringStage`, `KeywordExtractionStage` | 20 Newsgroups | ~18,000 posts, 20 topics |
| `02_sentiment_reviews_dedup_pipeline.py` | Sentiment classification + near-duplicate detection | `TextClassificationStage`, `TfidfVectorizeStage` (consumed by a custom dedup step, not by the classifier) | UCI Sentiment Labelled Sentences | ~3,000 sentences |
| `03_sms_spam_triage_pipeline.py` | Spam triage with conditional stages | `TextClassificationStage`, `KeywordExtractionStage`, both gated with `run_if` | UCI SMS Spam Collection | 5,574 messages |

## Where to get the data

**20 Newsgroups** — no manual download needed, it's built into scikit-learn:
```python
from sklearn.datasets import fetch_20newsgroups

fetch_20newsgroups(subset="train")  # downloads + caches to ~/scikit_learn_data
```
Original source: http://qwone.com/~jason/20Newsgroups/

**UCI Sentiment Labelled Sentences** (Kotzias et al., 2015) — a zip
containing `amazon_cells_labelled.txt`, `imdb_labelled.txt`, and
`yelp_labelled.txt`, each `sentence<TAB>label` (1=positive, 0=negative):
```
https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
```
Also mirrored on Kaggle under the same name if you'd rather use a browser.

**UCI SMS Spam Collection** — 5,574 labelled SMS messages, one file
(`SMSSpamCollection`), tab-separated as `label<TAB>message`:
```
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
```
Also mirrored on Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Other good sources for building your own examples

- **Hugging Face Datasets** (https://huggingface.co/datasets) — thousands of
  NLP datasets loadable in one line via `datasets.load_dataset(...)`; good
  for classification, QA, and summarization tasks at any scale.
- **Kaggle Datasets** (https://www.kaggle.com/datasets) — huge breadth,
  including support-ticket, review, and social-media corpora that map
  naturally onto this pipeline's stage types.
- **UCI Machine Learning Repository** (https://archive.ics.uci.edu/datasets) —
  smaller, well-documented, classic datasets; good for quick demos like the
  two UCI sets used above.
- **data.gov / government open data** (https://data.gov) — e.g. the CFPB
  Consumer Complaint Database, useful for a larger-scale ticket-triage-style
  example than SMS spam.

## Running

```bash
python examples/01_newsgroups_topic_pipeline.py
python examples/02_sentiment_reviews_dedup_pipeline.py [path/to/sentiment_data_dir]
python examples/03_sms_spam_triage_pipeline.py [path/to/SMSSpamCollection]
```

Omit the `[path/to/...]` arguments to select from paths in `data/`.