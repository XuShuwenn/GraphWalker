

python3 kgqa_agent/scripts/evaluate_valid_paths_with_answers.py \
    --input kgqa_agent/data/conjunction/paths_with_info/info_14k_deduplicated_valid_paths_with_answers.json \
    --data-type conj \
    --model gpt-4o-mini \
    --threshold 9 \
    --threads 10 \
    --temperature 0.0