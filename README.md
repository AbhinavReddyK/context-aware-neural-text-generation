# Context-Aware Neural Text Generation

Comparing three transformer approaches to two NLG tasks — **summarization**
and **story generation** — built with PyTorch and HuggingFace Transformers:

1. **GPT-2** (124M, pretrained) — fine-tuned as a prompted causal LM.
2. **T5-small** (60M, pretrained) — fine-tuned as a seq2seq summarizer.
3. **A custom lightweight Transformer encoder-decoder** — built and trained
   from scratch (~12M params), aiming for much faster inference at some
   quality cost.

Evaluated with BLEU-4, ROUGE-L, BERTScore, inference latency, and a small
manual human-eval pass.

Data: small samples pulled from CNN/DailyMail (summarization) and TinyStories
(story generation) via HF `datasets` streaming, so the whole thing runs on a
laptop CPU in well under an hour. See the note at the bottom for how to scale
it up to the full datasets on a GPU.

## What's here

```
data/prepare_data.py                sample + save CNN/DailyMail + TinyStories
models/custom_encoder_decoder.py    from-scratch Transformer seq2seq
scripts/train_t5_summarization.py
scripts/train_gpt2_summarization.py
scripts/train_custom_summarization.py
scripts/evaluate_summarization.py   BLEU-4 / ROUGE-L / BERTScore + latency
scripts/train_story_generation.py
scripts/generate_story_samples.py
results/                            metrics.json + sample outputs
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/prepare_data.py

cd scripts
python train_t5_summarization.py
python train_gpt2_summarization.py
python train_custom_summarization.py
python evaluate_summarization.py     # -> ../results/metrics.json

python train_story_generation.py
python generate_story_samples.py     # -> ../results/story_samples.md
```

or just `bash scripts/run_all.sh`.

## Custom model

A small pre-norm `nn.Transformer` (3 encoder + 3 decoder layers, d_model=192,
4 heads, ~12M params, tied embeddings) trained from scratch on the task data,
using the GPT-2 tokenizer just as a fixed vocabulary (embeddings are
initialized from a random projection of GPT-2's pretrained embeddings, but no
GPT-2 transformer weights are used). The speed/quality tradeoff comes from
being a much smaller model, not from distillation.

## Results

Summarization, test split (30 held-out CNN/DailyMail articles), measured on
this machine:

| Model | Params | BLEU-4 | ROUGE-L | BERTScore F1 | Latency / example |
|---|---|---|---|---|---|
| GPT-2 (fine-tuned) | 124.4M | 4.12 | 0.184 | 0.760 | 813 ms |
| T5-small (fine-tuned) | 60.5M | 6.29 | 0.217 | 0.783 | 280 ms |
| Custom lightweight | 12.2M | 1.90 | 0.085 | 0.685 | 180 ms |

Custom model vs. GPT-2: **4.5x faster**, retains **90% of BERTScore F1**
(0.685 / 0.760). BLEU-4/ROUGE-L drop more than BERTScore does — see
"what the custom model actually learned" below for why.

Full numbers in [`results/metrics.json`](results/metrics.json), example
outputs in [`results/sample_outputs.md`](results/sample_outputs.md) and
[`results/story_samples.md`](results/story_samples.md).

### What the custom model actually learned

Worth being upfront about: on ~300 training examples, a from-scratch model
learns the *style* of a news summary (grammatical, "NEW: ...", plausible
entities) well before it learns to faithfully ground content in the source
article. So its summaries read fine but sometimes get facts/names wrong —
that's why BLEU-4/ROUGE-L (which need exact word overlap with the reference)
drop more than BERTScore (which rewards semantic/stylistic similarity). More
training data would close this gap; see "Notes on scale" below.

To make the from-scratch model trainable at all on this little data, its
embedding table is initialized from a random projection of GPT-2's
pretrained token embeddings rather than from scratch (see
`gpt2_projected_embedding_init` in `models/custom_encoder_decoder.py`) — with
random init and ~300 examples it collapsed to repeating one token. The
Transformer encoder/decoder itself is still fully custom and trained from
scratch; only the embedding initialization is borrowed.

### Human eval (quick, single-annotator)

A rough manual read of the outputs in `results/sample_outputs.md` and
`results/story_samples.md`, scored 1-5 on fluency / coherence / relevance —
this is a sanity check on a handful of examples, not a powered study:

| Model | Fluency | Coherence | Relevance |
|---|---|---|---|
| GPT-2 summarization | 4 | 4 | 3.5 |
| T5-small summarization | 4 | 4 | 4 |
| Custom summarization | 3 | 2.5 | 1.5 |
| GPT-2 story generation | 3 | 2.5 | 3 |
| Custom story generation | 2 | 1.5 | 1.5 |

Matches the automatic metrics: T5 ≈ GPT-2 > custom on content quality, and
both fine-tuned tasks show the repetition loops you'd expect from single-epoch
fine-tuning on ~40 examples (see the story samples for "very happy... very
happy" -style loops).

## Notes on scale

This runs on small (few-hundred-example) samples so it's reproducible on a
CPU in one sitting — everything reported was actually run, not estimated.
To scale up to the full datasets, bump the sample sizes in
`data/prepare_data.py` and run on a GPU; the training scripts already move
to `cuda` automatically when available.

## Skills used

Python, PyTorch, HuggingFace Transformers, NLP (seq2seq fine-tuning, BLEU-4 /
ROUGE-L / BERTScore, custom Transformer design).

## License

MIT — see [LICENSE](LICENSE).
