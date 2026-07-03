#!/usr/bin/env bash
set -euo pipefail

# Countdown EGGROLL training + HellaSwag forgetting probe (disjoint splits, no leakage).
# Every 5 epochs: countdown val (disjoint) + HellaSwag val (HF validation split).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
TIME_BUDGET="${TIME_BUDGET_SECONDS:-3600}"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

LOG="$LOG_DIR/qwen35_countdown_hellaswag_forgetting.log"
echo "=== Countdown + HellaSwag forgetting (${TIME_BUDGET}s) -> $LOG ==="

exec "$VENV_PYTHON" -m llm_experiments.general_do_evolution \
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
  --train-dataset-size 256 \
  --val-dataset-size 256 \
  --time-budget-seconds "$TIME_BUDGET" \
  --aux-validation-task hellaswag \
  --hellaswag-val-size 64 \
  --hellaswag-val-seed 42 \
  --wandb-name countdown_hellaswag_forgetting \
  2>&1 | tee "$LOG"
