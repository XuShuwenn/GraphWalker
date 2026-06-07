#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

# Load .env if present (supports KEY=VALUE lines)
if [ -f .env ]; then
    # export all variables defined in .env into the environment
    set -a
    . .env
    set +a
fi


python -m kgqa_agent.src.data_gen.traj_gen.traj_generator \
  --in-file kgqa_agent/data/conjunction/paths_with_info/info_valid_paths_with_answers_avg_ge_9.json \
  --traj-out kgqa_agent/data/conjunction/rajectories/traj.4o-mini.json \
  --limit 10000 \
  --workers 30 \
  --checkpoint-every 5 \
  --max-retries-per-path 3 \
  --model "gpt-4o-mini" \
  --temperature 0.6 \
  --max-tokens 8192

