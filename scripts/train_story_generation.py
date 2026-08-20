"""Lightweight fine-tune of GPT-2 and the custom encoder-decoder for story
continuation on a small TinyStories sample. This is the secondary,
qualitative task in the study (see README "Honest Scope" section) -- the
quantitative BLEU-4/ROUGE-L/BERTScore/latency comparison lives in
evaluate_summarization.py; this script just produces the two checkpoints
used by generate_story_samples.py.
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
from custom_encoder_decoder import CustomEncoderDecoder, gpt2_projected_embedding_init  # noqa: E402

from data_utils import CKPT_DIR, story_pairs

MAX_PROMPT_LEN = 48
MAX_CONT_LEN = 80
EPOCHS_GPT2 = 1
EPOCHS_CUSTOM = 20
BATCH_SIZE = 4


class GPT2StoryDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.examples = []
        max_len = MAX_PROMPT_LEN + MAX_CONT_LEN
        for prompt, continuation in pairs:
            prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_PROMPT_LEN)["input_ids"]
            cont_ids = tokenizer(continuation, truncation=True, max_length=MAX_CONT_LEN - 1)[
                "input_ids"
            ] + [tokenizer.eos_token_id]
            input_ids = (prompt_ids + cont_ids)[:max_len]
            labels = ([-100] * len(prompt_ids) + cont_ids)[:max_len]
            pad_len = max_len - len(input_ids)
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


class CustomStoryDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.examples = []
        pad_id, bos_id, eos_id = tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
        for prompt, continuation in pairs:
            src = tokenizer(
                prompt, truncation=True, max_length=MAX_PROMPT_LEN, padding="max_length"
            )["input_ids"]
            tgt_ids = tokenizer(continuation, truncation=True, max_length=MAX_CONT_LEN - 2)[
                "input_ids"
            ]
            decoder_input = ([bos_id] + tgt_ids)[:MAX_CONT_LEN]
            labels = (tgt_ids + [eos_id])[:MAX_CONT_LEN]
            pad_len = MAX_CONT_LEN - len(decoder_input)
            decoder_input += [pad_id] * pad_len
            labels += [-100] * (MAX_CONT_LEN - len(labels))
            self.examples.append(
                {
                    "src": torch.tensor(src),
                    "decoder_input": torch.tensor(decoder_input),
                    "labels": torch.tensor(labels),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train_gpt2(device):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    ds = GPT2StoryDataset(story_pairs("train"), tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS_GPT2):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        print(f"[gpt2-story] epoch {epoch+1}/{EPOCHS_GPT2} loss={total_loss/len(loader):.4f}")
    print(f"[gpt2-story] training time: {time.time()-t0:.1f}s")

    out_dir = os.path.join(CKPT_DIR, "gpt2_story")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[gpt2-story] saved -> {out_dir}")


def train_custom(device):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    ds = CustomStoryDataset(story_pairs("train"), tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)
    model = CustomEncoderDecoder(
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    with torch.no_grad():
        model.embed.weight.copy_(gpt2_projected_embedding_init(vocab_size, model.d_model).to(device))
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    t0 = time.time()
    for epoch in range(EPOCHS_CUSTOM):
        total_loss = 0.0
        for batch in loader:
            src = batch["src"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            labels = batch["labels"].to(device)
            logits = model(src, decoder_input)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        print(f"[custom-story] epoch {epoch+1}/{EPOCHS_CUSTOM} loss={total_loss/len(loader):.4f}")
    print(f"[custom-story] training time: {time.time()-t0:.1f}s")

    out_dir = os.path.join(CKPT_DIR, "custom_story")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    tokenizer.save_pretrained(out_dir)
    print(f"[custom-story] saved -> {out_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_gpt2(device)
    train_custom(device)


if __name__ == "__main__":
    main()
