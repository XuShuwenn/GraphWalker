#!/usr/bin/env bash
# Small-sample run for synthesize_information on test_1k paths
# Usage: bash scripts/info_syn.sh
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

# Inputs/outputs
PATHS="$ROOT/kgqa_agent/data/3-5hop/test_500k/paths/st128.filtered.json"
QA="$ROOT/kgqa_agent/data/3-5hop/test_500k/qa/st128.filtered.gemini.jsonl"
OUT="$ROOT/kgqa_agent/data/3-5hop/test_500k/paths_with_info/info_st128.filtered.json"

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
  --max-num 50 \
  --resume \
  
echo "Done. Output -> $OUT"