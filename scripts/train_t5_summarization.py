"""Fine-tunes T5-small for summarization on the small CNN/DailyMail sample."""
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from data_utils import CKPT_DIR, summarization_pairs

MODEL_NAME = "t5-small"
MAX_SRC_LEN = 192
MAX_TGT_LEN = 64
EPOCHS = 1
BATCH_SIZE = 4
LR = 5e-5
OUT_DIR = os.path.join(CKPT_DIR, "t5_summarization")


class SumDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tok = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        article, summary = self.pairs[idx]
        src = self.tok(
            "summarize: " + article,
            truncation=True,
            max_length=MAX_SRC_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = self.tok(
            summary,
            truncation=True,
            max_length=MAX_TGT_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": labels,
        }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
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
        print(f"[t5] epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f}")
    print(f"[t5] training time: {time.time()-t0:.1f}s")

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[t5] saved checkpoint -> {OUT_DIR}")


if __name__ == "__main__":
    main()
