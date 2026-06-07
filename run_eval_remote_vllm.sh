#!/usr/bin/env bash
set -euo pipefail

# Remote vLLM runner: call remote vLLM server via API
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATASET="$REPO_ROOT/kgqa_agent/src/eval/datasets/cwq_test.json"
DATASET_TYPE="cwq"
OUTPUT_DIR="$REPO_ROOT/eval_results/remote/cwq/graphwalker-7b"

# Remote vLLM server configuration
REMOTE_SERVER_URL="${REMOTE_VLLM_URL:-http://localhost:22245/v1}"
MODEL_NAME="${REMOTE_MODEL_NAME:-graphwalker-7b}"
API_KEY="${REMOTE_VLLM_API_KEY:-}"

MODEL_KIND="local"  # or "vllm"
MODEL_CONFIG=$(printf '{
  "model_path": "%s",
  "base_url": "%s",
  "api_key": "%s",
  "temperature": 0.6,
  "max_tokens": 4096
}' "$MODEL_NAME" "$REMOTE_SERVER_URL" "$API_KEY")


LIMIT=4000
NUM_WORKERS=20
KG_SERVER_URL="http://localhost:18890"
MAX_CALLS=10
KG_TOP_K=20
FULL_LIST=false.  # whether to use full predicate whitelist for filtering


ARGS=(
  --dataset "$DATASET"
  --dataset-type "$DATASET_TYPE"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL_KIND"
  --limit "$LIMIT"
  --num-workers "$NUM_WORKERS"
  --model-config "$MODEL_CONFIG"
  --kg-top-k "$KG_TOP_K"
  --kg-server-url "$KG_SERVER_URL"
  --max-calls "$MAX_CALLS"
)

# Function to check if remote server is healthy and can process requests
check_server_health() {
  local server_url="$1"
  local model_name="$2"
  local api_key="$3"
  
  # Build curl command with optional API key
  local curl_args=(-s -f --max-time 10)
  if [ -n "$api_key" ]; then
    curl_args+=(-H "Authorization: Bearer $api_key")
  fi
  
  # Check /v1/models endpoint
  local models_response
  models_response=$(curl "${curl_args[@]}" "$server_url/models" 2>/dev/null) || return 1
  
  # Check if response contains the model name (best case: model is loaded)
  if echo "$models_response" | grep -q "$model_name" 2>/dev/null; then
    return 0  # Server is healthy and model is available
  fi
  
  # If model name not found, check if response is valid JSON (server might be up but model not loaded)
  if echo "$models_response" | python3 -m json.tool >/dev/null 2>&1; then
    # Server is up and responding correctly (even if model not loaded yet)
    # We'll consider it healthy if server responds with valid JSON
    return 0
  fi
  
  return 1  # Invalid response
}

# Wait for server to be ready, checking every 1000 seconds
CHECK_INTERVAL=60
echo "Waiting for remote vLLM server to be ready..."
echo "Server URL: $REMOTE_SERVER_URL"
echo "Model: $MODEL_NAME"
echo "Check interval: ${CHECK_INTERVAL} seconds"
echo ""

while true; do
  current_time=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$current_time] Checking server health..."
  
  if check_server_health "$REMOTE_SERVER_URL" "$MODEL_NAME" "$API_KEY"; then
    echo "[$current_time] ✓ Server is healthy and ready!"
    echo "[$current_time] Starting evaluation..."
    echo ""
    break
  else
    echo "[$current_time] ✗ Server is not ready yet. Waiting ${CHECK_INTERVAL} seconds before next check..."
    sleep "$CHECK_INTERVAL"
  fi
done

echo "Running with remote vLLM server at: $REMOTE_SERVER_URL"
echo "Model: $MODEL_NAME"
echo "Args: ${ARGS[*]}"
echo ""
echo "API Configuration:"
echo "  Main API - REMOTE_VLLM_URL: $REMOTE_SERVER_URL"
echo "  Main API - REMOTE_VLLM_API_KEY set: ${API_KEY:+yes}"
echo "  Filter API - FILTER_API_KEY set: ${FILTER_API_KEY:+yes}"
echo "  Filter API - FILTER_API_URL set: ${FILTER_API_URL:+yes}"
echo "  Filter API - FILTER_MODEL set: ${FILTER_MODEL:-gpt-4o-mini (default)}"
echo ""
mkdir -p "$REPO_ROOT/logs"

PYTHONPATH=. python3 -u -m kgqa_agent.scripts.run_kg_eval "${ARGS[@]}" 2>"$REPO_ROOT/logs/remote/remote_vllm_$(date +%Y%m%d_%H%M%S).log"
