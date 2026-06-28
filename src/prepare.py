"""Download the Nepali subset of XL-Sum and write train/val/test JSONL splits.

Run locally (no GPU needed):
    python src/prepare.py

We download the data file directly instead of using
load_dataset('csebuetnlp/xlsum', ...), because that relies on a loading script
which datasets 4.0+ no longer supports.

Each output line: {"text": <article>, "summary": <summary>, "title": <title>}
"""
import json
import tarfile
from pathlib import Path

import requests

URL = "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/nepali_XLSum_v2.0.tar.bz2"
SPLIT = {"train": "train", "val": "validation", "test": "test"}

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
ARCHIVE = Path("data/raw/nepali_XLSum_v2.0.tar.bz2")
ARCHIVE.parent.mkdir(parents=True, exist_ok=True)


def main():
    if not ARCHIVE.exists():
        print(f"Downloading {URL} ...")
        resp = requests.get(URL, timeout=120)
        resp.raise_for_status()
        ARCHIVE.write_bytes(resp.content)

    with tarfile.open(ARCHIVE, "r:bz2") as t:
        for m in t.getmembers():
            if not m.name.endswith(".jsonl"):
                continue
            tag = m.name.split("_")[-1].replace(".jsonl", "")  # train / val / test
            split = SPLIT[tag]
            rows = [
                json.loads(line)
                for line in t.extractfile(m).read().decode("utf-8").splitlines()
                if line.strip()
            ]
            path = OUT / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(
                        {"text": r["text"], "summary": r["summary"], "title": r["title"]},
                        ensure_ascii=False,
                    ) + "\n")
            print(f"{split}: {len(rows)} examples -> {path}")


if __name__ == "__main__":
    main()
