import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_jsonl(name):
    path = os.path.join(DATA_DIR, name)
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarization_pairs(split):
    rows = load_jsonl(f"summarization_{split}.jsonl")
    return [(r["article"], r["summary"]) for r in rows]


def story_pairs(split):
    rows = load_jsonl(f"stories_{split}.jsonl")
    return [(r["prompt"], r["continuation"]) for r in rows]
