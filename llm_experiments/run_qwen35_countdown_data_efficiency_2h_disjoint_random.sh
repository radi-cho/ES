#!/usr/bin/env bash
set -euo pipefail

# EGGROLL data-efficiency: disjoint train/val, random 8 train prompts/epoch, 2h/GPU each.
# Train pools: D=256 vs D=8.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
TIME_BUDGET="${TIME_BUDGET_SECONDS:-7200}"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

run_case() {
  local train_d="$1"
  local log="$LOG_DIR/qwen35_countdown_eggroll_D${train_d}_2h_disjoint_random.log"
  echo "=== EGGROLL D${train_d} disjoint+random (${TIME_BUDGET}s) -> $log ==="
  "$VENV_PYTHON" -m llm_experiments.general_do_evolution \
    --task countdown_chat \
    --noiser eggroll \
    --model-choice q35_2B \
    --rwkv-type Qwen35RWKV \
    --parallel-generations-per-gpu 64 \
    --generations-per-prompt 8 \
    --sigma 1e-3 \
    --lr-scale 0.2 \
    --seed 0 \
    --temperature 0.0 \
    --parallel-validations 64 \
    --validation-iterations 10 \
    --thinking-length 1024 \
    --answer-length 0 \
    --validate-every 5 \
    --train-dataset-size "$train_d" \
    --val-dataset-size 256 \
    --time-budget-seconds "$TIME_BUDGET" \
    --random-train-prompts \
    --wandb-name "D${train_d}_2h_disjoint_rand" \
    2>&1 | tee "$log"
  echo "=== Finished EGGROLL D${train_d} (exit $?) ==="
}

run_case 256
run_case 8

echo "EGGROLL 2h disjoint+random runs (D256, D8) complete."
