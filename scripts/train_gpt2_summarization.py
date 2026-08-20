"""Fine-tunes GPT-2 (124M) for summarization on the small CNN/DailyMail sample.

Uses the standard causal-LM prompt format: the article and summary are
concatenated with a delimiter, and loss is computed only on the summary +
eos tokens (the article acts as a prompt, not a training target).
"""
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from data_utils import CKPT_DIR, summarization_pairs

MODEL_NAME = "gpt2"
MAX_LEN = 256
EPOCHS = 1
BATCH_SIZE = 4
LR = 5e-5
OUT_DIR = os.path.join(CKPT_DIR, "gpt2_summarization")

DELIM = "\nTL;DR:\n"


class SumDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.examples = []
        for article, summary in pairs:
            prompt_ids = tokenizer(article + DELIM, truncation=True, max_length=MAX_LEN - 64)[
                "input_ids"
            ]
            target_ids = tokenizer(summary, truncation=True, max_length=64)["input_ids"] + [
                tokenizer.eos_token_id
            ]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            input_ids = input_ids[:MAX_LEN]
            labels = labels[:MAX_LEN]
            pad_len = MAX_LEN - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "labels": torch.tensor(labels),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_pairs = summarization_pairs("train")
    train_ds = SumDataset(train_pairs, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        print(f"[gpt2] epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f}")
    print(f"[gpt2] training time: {time.time()-t0:.1f}s")

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[gpt2] saved checkpoint -> {OUT_DIR}")


if __name__ == "__main__":
    main()
