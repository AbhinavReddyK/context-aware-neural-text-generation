"""Trains the lightweight custom encoder-decoder from scratch for summarization.

Uses the GPT-2 BPE tokenizer purely as a fixed, off-the-shelf vocabulary
(no pretrained GPT-2 weights are used) so all three models are compared on
the same token space.
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
from custom_encoder_decoder import CustomEncoderDecoder, gpt2_projected_embedding_init  # noqa: E402

from data_utils import CKPT_DIR, summarization_pairs

MAX_SRC_LEN = 192
MAX_TGT_LEN = 64
EPOCHS = 15
BATCH_SIZE = 4
LR = 3e-4
OUT_DIR = os.path.join(CKPT_DIR, "custom_summarization")


class SumDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.examples = []
        pad_id = tokenizer.pad_token_id
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        for article, summary in pairs:
            src = tokenizer(
                article,
                truncation=True,
                max_length=MAX_SRC_LEN,
                padding="max_length",
            )["input_ids"]
            tgt_ids = tokenizer(summary, truncation=True, max_length=MAX_TGT_LEN - 2)[
                "input_ids"
            ]
            decoder_input = [bos_id] + tgt_ids
            labels = tgt_ids + [eos_id]
            pad_len = MAX_TGT_LEN - len(decoder_input)
            decoder_input = decoder_input + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            self.examples.append(
                {
                    "src": torch.tensor(src),
                    "decoder_input": torch.tensor(decoder_input[:MAX_TGT_LEN]),
                    "labels": torch.tensor(labels[:MAX_TGT_LEN]),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token  # gpt2 has no dedicated bos; reuse eos id as bos

    train_pairs = summarization_pairs("train")
    train_ds = SumDataset(train_pairs, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)
    model = CustomEncoderDecoder(
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    with torch.no_grad():
        model.embed.weight.copy_(
            gpt2_projected_embedding_init(vocab_size, model.d_model).to(device)
        )
    print(f"[custom] parameters: {model.num_parameters():,}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            src = batch["src"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            labels = batch["labels"].to(device)

            logits = model(src, decoder_input)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        print(f"[custom] epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f}")
    print(f"[custom] training time: {time.time()-t0:.1f}s")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[custom] saved checkpoint -> {OUT_DIR}")


if __name__ == "__main__":
    main()
