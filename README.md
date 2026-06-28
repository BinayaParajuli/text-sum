# Nepali Abstractive Summarization with Multilingual Transformers

Masters project. Fine-tune multilingual transformers (mT5, mBART, IndicBART) with
LoRA for abstractive summarization of Nepali news, extending Dhakal & Baral (2024,
arXiv:2409.19566).

## Goal

1. **Baseline** — reproduce headline generation with mT5/mBART + LoRA.
2. **Extend** — full short summaries (not just headlines), add IndicBART, better eval.
3. **Evaluate** — ROUGE + BERTScore + human evaluation.

## Where things run

- **Local Mac (8GB, Intel):** data scraping, cleaning, analysis, writing. NO training.
- **Google Colab (free T4 GPU):** all model fine-tuning and evaluation.

## Structure

```
data/raw/         scraped news (article + headline)
data/processed/   cleaned train/val/test splits
src/              reusable python (scraper, cleaning, dataset prep)
notebooks/        Colab notebooks for training/eval
results/          metrics, sample outputs
```

## Setup (local)

Python 3.14 is too new for PyTorch. Use Python 3.11:

```bash
brew install python@3.11
/usr/local/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

## Workflow

1. `python src/scrape.py`     -> data/raw/*.jsonl
2. `python src/prepare.py`    -> data/processed/{train,val,test}.jsonl
3. Upload processed data to Google Drive, open notebooks/ in Colab, train.
