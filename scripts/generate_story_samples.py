"""Produces qualitative story continuations from GPT-2 and the custom model
for a handful of held-out prompts. Writes results/story_samples.md.
"""
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
from custom_encoder_decoder import CustomEncoderDecoder  # noqa: E402

from data_utils import CKPT_DIR, RESULTS_DIR, story_pairs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 8


def main():
    test_pairs = story_pairs("test")[:N_SAMPLES]

    gpt2_dir = os.path.join(CKPT_DIR, "gpt2_story")
    gpt2_tok = GPT2TokenizerFast.from_pretrained(gpt2_dir)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_dir).to(DEVICE).eval()

    custom_dir = os.path.join(CKPT_DIR, "custom_story")
    custom_tok = GPT2TokenizerFast.from_pretrained(custom_dir)
    custom_tok.pad_token = custom_tok.eos_token
    custom_tok.bos_token = custom_tok.eos_token
    custom_model = CustomEncoderDecoder(
        vocab_size=custom_tok.vocab_size + len(custom_tok.added_tokens_encoder),
        pad_token_id=custom_tok.pad_token_id,
    )
    custom_model.load_state_dict(
        torch.load(os.path.join(custom_dir, "model.pt"), map_location=DEVICE)
    )
    custom_model.to(DEVICE).eval()

    lines = ["# Story generation samples (TinyStories, held-out prompts)\n"]
    for prompt, reference in test_pairs:
        lines.append(f"\n## Prompt\n{prompt}\n")
        lines.append(f"**Reference continuation:** {reference}\n")

        with torch.no_grad():
            inp = gpt2_tok(prompt, return_tensors="pt").to(DEVICE)
            out = gpt2_model.generate(
                **inp, max_new_tokens=40, do_sample=False, pad_token_id=gpt2_tok.pad_token_id
            )
            gpt2_out = gpt2_tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        lines.append(f"**GPT-2 (fine-tuned):** {gpt2_out.strip()}\n")

        with torch.no_grad():
            src = custom_tok(prompt, truncation=True, max_length=48, return_tensors="pt")[
                "input_ids"
            ].to(DEVICE)
            out = custom_model.generate(
                src,
                bos_token_id=custom_tok.eos_token_id,
                eos_token_id=custom_tok.eos_token_id,
                max_new_tokens=40,
            )
            custom_out = custom_tok.decode(out[0], skip_special_tokens=True)
        lines.append(f"**Custom lightweight model:** {custom_out.strip()}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "story_samples.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
