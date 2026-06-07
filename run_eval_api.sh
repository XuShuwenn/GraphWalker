#!/usr/bin/env bash
set -euo pipefail

# Hardcoded runner: load .env (all vars) and run run_kg_eval.py with built-in parameters
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/kgqa_agent/src/eval:${PYTHONPATH:-}"

# Model configuration
MODEL_KIND="api"
MODEL_NAME="gpt-4o-mini"
MODEL_CONFIG=$(printf '{"model":"%s","temperature":0.6,"max_tokens":4096}' "$MODEL_NAME")

# Common evaluation parameters
LIMIT=10000
NUM_WORKERS=10
KG_SERVER_URL="http://localhost:18890"
MAX_CALLS=10
KG_TOP_K=20
FULL_LIST=false

# Dataset configurations: name, file, dataset_type
DATASETS=(
  "cwq:kgqa_agent/src/eval/datasets/cwq_test.json:cwq"
  "webqsp:kgqa_agent/src/eval/datasets/webqsp_test.json:webqsp"
  "GWBench:kgqa_agent/src/eval/datasets/GWBench.json:GWBench"
  "grailqa:kgqa_agent/src/eval/datasets/graliqa_test_1000.json:grailqa"
)

echo "API Configuration:"
echo "  Main API - OPENAI_API_KEY set: ${OPENAI_API_KEY:+yes}"
echo "  Main API - OPENAI_BASE_URL set: ${OPENAI_BASE_URL:+yes}"
echo "  Filter API - FILTER_API_KEY set: ${FILTER_API_KEY:+yes}"
echo "  Filter API - FILTER_API_URL set: ${FILTER_API_URL:+yes}"
echo "  Filter API - FILTER_MODEL set: ${FILTER_MODEL:-gpt-4o-mini (default)}"
echo "  Model: $MODEL_NAME"
echo ""
mkdir -p "$REPO_ROOT/logs/api"

# Loop through all datasets
for DATASET_CONFIG in "${DATASETS[@]}"; do
  # Parse dataset configuration
  IFS=':' read -r NAME FILE TYPE <<< "$DATASET_CONFIG"
  
  echo ""
  echo "========================================="
  echo "Evaluating dataset: $NAME"
  echo "File: $FILE"
  echo "Type: $TYPE"
  echo "========================================="
  
  # Set output directory and path for this dataset
  OUTPUT_DIR="$REPO_ROOT/eval_results/api/Rebuttal/filter/BestN$NAME/$MODEL_NAME"
  OUTPUT_PATH="$OUTPUT_DIR/${MODEL_NAME}_${NAME}.json"
  
  # Build arguments for this dataset
  ARGS=(
    --dataset "$REPO_ROOT/$FILE"
    --dataset-type "$TYPE"
    --output-dir "$OUTPUT_DIR"
    --output-path "$OUTPUT_PATH"
    --model "$MODEL_KIND"
    --limit "$LIMIT"
    --num-workers "$NUM_WORKERS"
    --model-config "$MODEL_CONFIG"
    --kg-top-k "$KG_TOP_K"
    --kg-server-url "$KG_SERVER_URL"
    --max-calls "$MAX_CALLS"
  )
  
  LOG_FILE="$REPO_ROOT/logs/api/${NAME}_$(date +%Y%m%d_%H%M%S).log"
  
  # Run evaluation
  echo "Running evaluation for $NAME..."
  echo "Output: $OUTPUT_PATH"
  echo "Log: $LOG_FILE"
  echo ""
  
  PYTHONPATH=. python3 -u -m kgqa_agent.scripts.run_kg_eval "${ARGS[@]}" 2>"$LOG_FILE"
  
  # Check if evaluation succeeded
  if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully completed evaluation for $NAME"
  else
    echo ""
    echo "✗ Evaluation failed for $NAME"
    echo "Check log file: $LOG_FILE"
    exit 1
  fi
done

echo ""
echo "========================================="
echo "All dataset evaluations completed!"
echo "========================================="

