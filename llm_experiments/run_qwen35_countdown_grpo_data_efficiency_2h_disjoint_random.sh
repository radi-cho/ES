#!/usr/bin/env bash
set -euo pipefail

# GRPO data-efficiency: disjoint train/val, random train prompts/epoch, 2h/GPU each.
# Train pools: D=256 vs D=8. Population = POPULATION_SIZE (default 256 = 8 prompts × 32 gen).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false

TIME_BUDGET="${TIME_BUDGET_SECONDS:-7200}"
POPULATION_SIZE="${POPULATION_SIZE:-256}"
GENERATIONS_PER_PROMPT="${GENERATIONS_PER_PROMPT:-$((POPULATION_SIZE / 8))}"
PARALLEL_VAL="${PARALLEL_VALIDATIONS:-64}"
VAL_ITERATIONS="${VALIDATION_ITERATIONS:-10}"
GRPO_MAX_SEQ="${GRPO_MAX_SEQ_LEN:-64}"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

run_case() {
  local train_d="$1"
  local log="$LOG_DIR/qwen35_countdown_grpo_D${train_d}_2h_disjoint_random_bs${POPULATION_SIZE}.log"
  echo "=== GRPO D${train_d} disjoint+random pop=${POPULATION_SIZE} gen/prompt=${GENERATIONS_PER_PROMPT} (${TIME_BUDGET}s) -> $log ==="
  set +e
  "$VENV_PYTHON" -u -m llm_experiments.do_grpo \
    --task countdown_chat \
    --model-choice q35_2B \
    --rwkv-type Qwen35RWKV \
    --parallel-generations-per-gpu "$POPULATION_SIZE" \
    --generations-per-prompt "$GENERATIONS_PER_PROMPT" \
    --lr 2e-6 \
    --train-temp 0.6 \
    --clip-eps 0.2 \
    --num-minibatches 8 \
    --freeze-nonlora \
    --rank 1 \
    --grpo-max-seq-len "$GRPO_MAX_SEQ" \
    --seed 0 \
    --parallel-validations "$PARALLEL_VAL" \
    --validation-iterations "$VAL_ITERATIONS" \
    --thinking-length 1024 \
    --answer-length 0 \
    --validate-every 5 \
    --train-dataset-size "$train_d" \
    --val-dataset-size 256 \
    --time-budget-seconds "$TIME_BUDGET" \
    --random-train-prompts \
    --initial-eval-examples 0 \
    --num-epochs 9999 \
    --wandb-name "D${train_d}_2h_disjoint_rand" \
    2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "=== Finished GRPO D${train_d} (exit ${rc}) ==="
  local latest
  latest=$(ls -td "$REPO_ROOT"/countdown_chat_grpo_*trainD="${train_d}"* 2>/dev/null | head -1 || true)
  if [[ -n "$latest" && -f "$latest/validation.csv" ]]; then
    "$VENV_PYTHON" -m llm_experiments.check_validation_health "$latest/validation.csv"
  fi
  return "$rc"
}

run_case 256
run_case 8

echo "GRPO 2h disjoint+random runs (D256, D8) complete."
