#!/bin/bash

# GraphWalker Training Script for Qwen2.5-3B
# Adapted from slime/examples/search-r1/run_qwen2.5_3B.sh

# ============== Logging Setup ==============
# Set up log file with timestamp
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Redirect all output (stdout and stderr) to both terminal and log file
# This ensures all output is saved to log file while still displaying on terminal
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

# Function to log with timestamp (after redirection, so it also goes to log)
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_with_timestamp "========================================="
log_with_timestamp "GraphWalker Training Started"
log_with_timestamp "Log file: ${LOG_FILE}"
log_with_timestamp "========================================="

# Clean up any existing processes
log_with_timestamp "Cleaning up existing processes..."
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

set -ex

# Load .env file if present (for OPENAI_API_KEY, FILTER_API_KEY, etc.)
[ -f "${SCRIPT_DIR}/.env" ] && set -a && source "${SCRIPT_DIR}/.env" && set +a

# Prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

# ============== Checkpoint Configuration ==============
CKPT_ARGS=(
   --hf-checkpoint /usercache/huggingface/Qwen2.5-3B-Instruct/
   --ref-load /Graphwalker/slime/models/qwen2.5-3b-instruct_torch_dist
   # --load /root/Qwen2.5-3B_slime/
   --save /Graphwalker/slime/saves/qwen2.5-3b-instruct_grpo/
   --save-interval 50
)
# ============== Data Configuration ==============
# Note: Run scripts/prepare_data.py first to generate the prepared dataset
ROLLOUT_ARGS=(
   --prompt-data ${SCRIPT_DIR}/datasets/cwq_train_prepared.jsonl
   --input-key question          # Maps to sample.prompt (original question, plain text)
   --label-key metadata          # We extract answers from metadata in reward_func
   --metadata-key metadata       # Loads topic_entity and answers

   # --apply-chat-template  
   
   --rollout-shuffle
   --num-rollout 1000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-num-gpus 4
   --rollout-sample-timeout-sec 900
   --rollout-max-response-len 512
   --rollout-temperature 1
   --num-steps-per-rollout 2
   # --global-batch-size 128
   # --sglang-max-running-requests 8
   # --partial-rollout
   # --mask-offpolicy-in-partial-rollout

   # over_sampling_batch_size >= rollout_batch_size; 8 组=32 样本 in-flight, ~8/引擎
   --over-sampling-batch-size 16
   # GraphWalker 数据源: get_samples 里将 metadata(JSON 字符串) 转成 dict,避免 slime abort() 报错
   --data-source-path graphwalker_data_source.GraphWalkerDataSource
   # --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

# ============== Evaluation Configuration ==============
# EVAL_ARGS=(
   # --eval-interval 5
   # --eval-prompt-data cwq_test /root/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:300]
   # --eval-prompt-data nq_test /root/nq_search/test.parquet
   # --eval-input-key prompt
   # --eval-label-key reward_model
   # --n-samples-per-eval-prompt 4
   # --balance-data 
# )



# ============== Performance Configuration ==============
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   # partial-rollout 可能产生更长轨迹; 降低以避训练阶段 OOM (上次 24576 在 logits.clone 处 OOM)
   --max-tokens-per-gpu 8192
)

# ============== GRPO Configuration ==============
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

# ============== Optimizer Configuration ==============
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ============== Wandb Configuration ==============
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group graphwalker_qwen2.5-3B
)

# ============== SGLang Configuration ==============
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # 4 卡 colocate: 略降 SGLang 占显存，给同卡后续训练 backward 留余量
   --sglang-mem-fraction-static 0.7
)

# ============== Misc Configuration ==============
MISC_ARGS=(
   # Default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # Should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # Need to comment this when using model with MLA
   --attention-backend flash
)

# ============== Custom Functions for GraphWalker ==============
CUSTOM_ARGS=(
   # GraphWalker's custom generate function (multi-turn KG interaction)
   --custom-generate-function-path generate_with_search.generate
   # GraphWalker's custom reward function (EM + format reward)
   --custom-rm-path generate_with_search.reward_func
   # TIS-related args, recommended to enable when using TIS
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# ============== Launch ==============

# Set KG server URL (remote Virtuoso SPARQL endpoint)
export KG_SERVER_URL="http://localhost:18890"

# Launch the master node of ray in container
log_with_timestamp "Starting Ray cluster..."
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats
log_with_timestamp "Ray cluster started at ${MASTER_ADDR}"

# Configure runtime environment (single line to avoid shell splitting the ray command)
# IMPORTANT: Include SCRIPT_DIR in PYTHONPATH for graphwalker imports
# Note: FILTER_API_KEY and FILTER_API_URL are optional - if not set, will fallback to OPENAI_API_KEY/OPENAI_BASE_URL
RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"/root/Megatron-LM/:${SCRIPT_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"KG_SERVER_URL\":\"http://localhost:18890\",\"FILTER_API_KEY\":\"${FILTER_API_KEY:-}\",\"FILTER_API_URL\":\"${FILTER_API_URL:-}\",\"FILTER_MODEL\":\"${FILTER_MODEL:-Qwen-2.5-7B-Instruct}\",\"OPENAI_API_KEY\":\"${OPENAI_API_KEY:-}\",\"OPENAI_BASE_URL\":\"${OPENAI_BASE_URL:-}\"}}"

log_with_timestamp "========================================="
log_with_timestamp "Submitting training job to Ray..."
log_with_timestamp "Ray address: http://127.0.0.1:8265"
log_with_timestamp "Log file: ${LOG_FILE}"
log_with_timestamp "========================================="

# Submit job to ray
# NOTE: This script must be run from the slime/ directory (parent of examples/)
# so that train.py can be found by Ray job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

# Note: All output (stdout and stderr) is automatically saved to:
# ${LOG_FILE}
# You can monitor the log file in real-time with: tail -f ${LOG_FILE}
log_with_timestamp "Training job submitted. Log file: ${LOG_FILE}"
log_with_timestamp "Monitor logs with: tail -f ${LOG_FILE}"
