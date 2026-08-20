"""Evaluates the three fine-tuned summarizers on the held-out test split.

Computes BLEU-4 (sacrebleu), ROUGE-L (rouge-score), BERTScore (F1, using a
lightweight scorer model for CPU feasibility), and per-example greedy-decoding
latency (single example per batch, CPU, warm model) so inference speed
numbers are measured, not asserted.

Writes results/metrics.json and results/sample_outputs.md.
"""
import json
import os
import sys
import time

import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
from custom_encoder_decoder import CustomEncoderDecoder  # noqa: E402

from data_utils import CKPT_DIR, RESULTS_DIR, summarization_pairs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERTSCORE_MODEL = "distilbert-base-uncased"  # lightweight scorer, CPU-feasible
DELIM = "\nTL;DR:\n"
MAX_SRC_LEN = 192
MAX_NEW_TOKENS = 32


def load_t5():
    path = os.path.join(CKPT_DIR, "t5_summarization")
    tok = T5TokenizerFast.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path).to(DEVICE).eval()
    return tok, model


def load_gpt2():
    path = os.path.join(CKPT_DIR, "gpt2_summarization")
    tok = GPT2TokenizerFast.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path).to(DEVICE).eval()
    return tok, model


def load_custom():
    path = os.path.join(CKPT_DIR, "custom_summarization")
    tok = GPT2TokenizerFast.from_pretrained(path)
    tok.pad_token = tok.eos_token
    tok.bos_token = tok.eos_token
    model = CustomEncoderDecoder(
        vocab_size=tok.vocab_size + len(tok.added_tokens_encoder),
        pad_token_id=tok.pad_token_id,
    )
    model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=DEVICE))
    model.to(DEVICE).eval()
    return tok, model


@torch.no_grad()
def gen_t5(tok, model, article):
    inp = tok("summarize: " + article, truncation=True, max_length=MAX_SRC_LEN, return_tensors="pt").to(
        DEVICE
    )
    t0 = time.perf_counter()
    out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, num_beams=1, do_sample=False)
    dt = time.perf_counter() - t0
    return tok.decode(out[0], skip_special_tokens=True).strip(), dt


@torch.no_grad()
def gen_gpt2(tok, model, article):
    prompt = article + DELIM
    inp = tok(prompt, truncation=True, max_length=MAX_SRC_LEN, return_tensors="pt").to(DEVICE)
    t0 = time.perf_counter()
    out = model.generate(
        **inp,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    dt = time.perf_counter() - t0
    gen_ids = out[0][inp["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip(), dt


@torch.no_grad()
def gen_custom(tok, model, article):
    src = tok(article, truncation=True, max_length=MAX_SRC_LEN, return_tensors="pt")["input_ids"].to(
        DEVICE
    )
    t0 = time.perf_counter()
    out = model.generate(
        src, bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, max_new_tokens=MAX_NEW_TOKENS
    )
    dt = time.perf_counter() - t0
    return tok.decode(out[0], skip_special_tokens=True).strip(), dt


def compute_text_metrics(preds, refs):
    bleu = BLEU(effective_order=True)
    bleu_scores = [bleu.sentence_score(p, [r]).score for p, r in zip(preds, refs)]
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = [rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    _, _, bert_f1 = bert_score(preds, refs, model_type=BERTSCORE_MODEL, verbose=False)
    return {
        "bleu4": sum(bleu_scores) / len(bleu_scores),
        "rougeL": sum(rouge_l) / len(rouge_l),
        "bertscore_f1": bert_f1.mean().item(),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_pairs = summarization_pairs("test")
    articles = [a for a, _ in test_pairs]
    references = [s for _, s in test_pairs]

    models = {
        "gpt2_finetuned": (load_gpt2, gen_gpt2),
        "t5_small_finetuned": (load_t5, gen_t5),
        "custom_lightweight": (load_custom, gen_custom),
    }

    all_metrics = {}
    sample_lines = ["# Sample outputs (summarization test split)\n"]

    for name, (loader, gen_fn) in models.items():
        print(f"=== {name} ===")
        tok, model = loader()
        n_params = sum(p.numel() for p in model.parameters())

        preds, latencies = [], []
        for article in articles:
            pred, dt = gen_fn(tok, model, article)
            preds.append(pred)
            latencies.append(dt)

        text_metrics = compute_text_metrics(preds, references)
        avg_latency_ms = 1000 * sum(latencies) / len(latencies)

        all_metrics[name] = {
            "parameters": n_params,
            "avg_latency_ms_per_example": avg_latency_ms,
            **text_metrics,
        }
        print(json.dumps(all_metrics[name], indent=2))

        sample_lines.append(f"\n## {name}\n")
        for i in range(min(5, len(preds))):
            sample_lines.append(f"**Reference:** {references[i]}\n\n**Prediction:** {preds[i]}\n")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    baseline_latency = all_metrics["gpt2_finetuned"]["avg_latency_ms_per_example"]
    custom_latency = all_metrics["custom_lightweight"]["avg_latency_ms_per_example"]
    all_metrics["_derived"] = {
        "custom_speedup_vs_gpt2": baseline_latency / custom_latency,
        "custom_bertscore_retention_vs_gpt2": (
            all_metrics["custom_lightweight"]["bertscore_f1"]
            / all_metrics["gpt2_finetuned"]["bertscore_f1"]
        ),
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "sample_outputs.md"), "w") as f:
        f.write("\n".join(sample_lines))

    print("\n=== derived ===")
    print(json.dumps(all_metrics["_derived"], indent=2))
    print(f"\nWrote results/metrics.json and results/sample_outputs.md")


if __name__ == "__main__":
    main()
