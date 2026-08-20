#!/usr/bin/env bash
# Runs the full pipeline end to end: data prep -> fine-tune all 3 models on
# both tasks -> evaluate -> qualitative story samples.
set -euo pipefail
cd "$(dirname "$0")/.."

python data/prepare_data.py

cd scripts
python train_t5_summarization.py
python train_gpt2_summarization.py
python train_custom_summarization.py
python evaluate_summarization.py

python train_story_generation.py
python generate_story_samples.py

echo "Done. See results/metrics.json, results/sample_outputs.md, results/story_samples.md"
