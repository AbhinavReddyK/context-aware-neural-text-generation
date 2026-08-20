"""
Builds small, fixed-size samples for the two tasks used in this project:

  - summarization: CNN/DailyMail (3.0.0), article -> highlights
  - story generation: TinyStories, short prompt -> continuation

The samples are intentionally small (a few hundred examples per split) so the
whole pipeline (fine-tune 3 models x eval) runs in minutes on a CPU-only
laptop. Swap SUMMARY_* / STORY_* sizes below to scale up on a GPU box.

Usage:
    python data/prepare_data.py
"""
import json
import os
import random

from datasets import load_dataset

random.seed(13)

OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

SUMMARY_TRAIN, SUMMARY_VAL, SUMMARY_TEST = 300, 30, 30
STORY_TRAIN, STORY_VAL, STORY_TEST = 40, 8, 8

MAX_ARTICLE_CHARS = 800
MAX_SUMMARY_CHARS = 300
MAX_STORY_CHARS = 600


def dump(rows, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def build_summarization():
    def take(split, n):
        stream = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split, streaming=True)
        rows = []
        for ex in stream:
            article = ex["article"].strip()
            summary = ex["highlights"].strip().replace("\n", " ")
            if not article or not summary:
                continue
            rows.append(
                {
                    "article": article[:MAX_ARTICLE_CHARS],
                    "summary": summary[:MAX_SUMMARY_CHARS],
                }
            )
            if len(rows) == n:
                break
        return rows

    dump(take("train", SUMMARY_TRAIN), "summarization_train.jsonl")
    dump(take("validation", SUMMARY_VAL), "summarization_val.jsonl")
    dump(take("test", SUMMARY_TEST), "summarization_test.jsonl")


def build_stories():
    def take(split, n, skip=0):
        stream = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        rows = []
        skipped = 0
        for ex in stream:
            text = ex["text"].strip().replace("\n", " ")
            if len(text) < 200:
                continue
            if skipped < skip:
                skipped += 1
                continue
            cut = max(40, len(text) // 4)
            prompt, continuation = text[:cut], text[cut:MAX_STORY_CHARS]
            rows.append({"prompt": prompt, "continuation": continuation})
            if len(rows) == n:
                break
        return rows

    dump(take("train", STORY_TRAIN), "stories_train.jsonl")
    dump(take("validation", STORY_VAL), "stories_val.jsonl")
    dump(take("validation", STORY_TEST, skip=STORY_VAL), "stories_test.jsonl")


if __name__ == "__main__":
    build_summarization()
    build_stories()
