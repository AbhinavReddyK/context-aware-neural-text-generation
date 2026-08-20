"""
A small from-scratch Transformer encoder-decoder used as the "lightweight
custom model" in the comparison. It is intentionally tiny relative to
GPT-2 (124M) and T5-small (60M) so that inference-speed gains are a direct
consequence of parameter count / architecture, not a trick.

It's a standard pre-norm Transformer seq2seq (nn.TransformerEncoder /
nn.TransformerDecoder) with sinusoidal positional encoding and a
shared input/output embedding table, trained from scratch (no pretrained
weights) directly on the task data.

The only pretrained thing this model borrows is its *embedding
initialization*: with only a few hundred training examples there isn't
enough signal to learn word-level semantics from a random init in the time
available, so the embedding table is initialized from a random linear
projection of GPT-2's pretrained token embeddings (see
`gpt2_projected_embedding_init` below), then trained like everything else.
No GPT-2 transformer weights are used anywhere in this model.
"""
import math

import torch
import torch.nn as nn


def gpt2_projected_embedding_init(vocab_size: int, d_model: int, seed: int = 13) -> torch.Tensor:
    """Random-projects GPT-2's pretrained token embeddings down to d_model.

    A random projection is a cheap way to carry over *some* of the semantic
    structure of a well-trained 768-dim embedding space into a much smaller
    space (Johnson-Lindenstrauss-style), without pulling in any GPT-2
    transformer weights. It's just a smarter initialization -- the embedding
    table remains fully trainable afterwards.
    """
    from transformers import GPT2Model

    wte = GPT2Model.from_pretrained("gpt2").wte.weight.data  # (50257, 768)
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(wte.size(1), d_model, generator=gen) / (wte.size(1) ** 0.5)
    projected = wte @ proj  # (50257, d_model)

    if vocab_size > projected.size(0):
        extra = torch.randn(vocab_size - projected.size(0), d_model, generator=gen) * 0.02
        projected = torch.cat([projected, extra], dim=0)
    return projected[:vocab_size].contiguous()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CustomEncoderDecoder(nn.Module):
    """Lightweight seq2seq Transformer, config sized for CPU fine-tuning."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 1024,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def _make_masks(self, src, tgt):
        src_key_padding_mask = src == self.pad_token_id
        tgt_key_padding_mask = tgt == self.pad_token_id
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )
        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask

    def forward(self, src, tgt):
        """src: (B, S) encoder input ids. tgt: (B, T) decoder input ids (shifted right)."""
        src_kpm, tgt_kpm, tgt_mask = self._make_masks(src, tgt)

        src_emb = self.pos_enc(self.embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.embed(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        return self.lm_head(out)

    @torch.no_grad()
    def generate(self, src, bos_token_id, eos_token_id, max_new_tokens=64):
        """Greedy decoding with immediate-repeat blocking (no KV cache, kept simple
        on purpose). A model this small and this lightly trained easily falls into
        "the the the" loops under pure argmax decoding, so we forbid picking the
        same token twice in a row -- if argmax repeats the previous token, fall
        back to the next-best logit instead.
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(src, tgt)
            step_logits = logits[:, -1, :].clone()
            prev_token = tgt[:, -1]
            step_logits.scatter_(1, prev_token.unsqueeze(1), float("-inf"))
            next_token = step_logits.argmax(dim=-1, keepdim=True)
            next_token[done] = eos_token_id
            tgt = torch.cat([tgt, next_token], dim=1)
            done = done | (next_token.squeeze(-1) == eos_token_id)
            if done.all():
                break
        return tgt
