#!/usr/bin/env bash
set -euo pipefail


python kgqa_agent/src/data_gen/generate_questions.py \
  --input kgqa_agent/data/3-5hop/test_500k/paths/st128.filtered.json \
  --max-workers 20 \
  --model 'gemini-2.5-pro' \
  --temperature 0.7 \
  --max-tokens 768 \
  --limit 30000 \
  --output kgqa_agent/data/3-5hop/test_500k/qa/st128.filtered.gemini.jsonl
