set -euo pipefail

python3 kgqa_agent/src/data_gen/generate_conjunction_questions.py \
    --input kgqa_agent/data/conjunction/paths/conjunction_paths_100k_names.filtered.json \
    --output kgqa_agent/data/conjunction/qa/conjunction_qa_100k.jsonl \
    --model gemini-2.5-pro \
    --temperature 0.7 \
    --max-tokens 768 \
    --limit 25000 \
    --max-workers 20