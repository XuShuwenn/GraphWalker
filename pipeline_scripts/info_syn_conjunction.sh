#!/usr/bin/env bash
# Synthesize information for conjunction paths
# Usage: bash pipeline_scripts/info_syn_conjunction.sh
# Optional: export KG_URL=http://localhost:18890

set -euo pipefail

# Resolve repo root: prefer git top-level (if script is inside a git worktree);
# otherwise fall back to the script directory. Make ROOT readonly so it cannot
# be changed later in the script or by callers.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_GIT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT="$ROOT_GIT"
else
  ROOT="$SCRIPT_DIR"
fi
readonly ROOT
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Inputs/outputs for conjunction paths
PATHS="$ROOT/kgqa_agent/data/conjunction/paths/conjunction_paths_100k_filtered.json"
QA="$ROOT/kgqa_agent/data/conjunction/qa/conjunction_qa_100k.jsonl"
OUT="$ROOT/kgqa_agent/data/conjunction/paths_with_info/info_conjunction_paths_100k.json"

KG_URL="${KG_URL:-http://localhost:18890}"

# Ensure output directory exists
mkdir -p "$(dirname "$OUT")"

python -m kgqa_agent.src.data_gen.synthesize_information \
  --paths-file "$PATHS" \
  --qa-file "$QA" \
  --out-file "$OUT" \
  --kg-server-url "$KG_URL" \
  --save-every 10 \
  --relation-limit 20 \
  --entity-limit 30 \
  --max-num 50 \
  --resume \
  
echo "Done. Output -> $OUT"

