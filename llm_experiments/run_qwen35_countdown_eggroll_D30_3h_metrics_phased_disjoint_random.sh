#!/usr/bin/env bash
set -euo pipefail

# D30 phased 3x10 @ 1h/phase, 3h, log val + train fitness on all 30 + update norms.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
TIME_BUDGET="${TIME_BUDGET_SECONDS:-10800}"
PHASE_SECONDS="${PHASE_SECONDS:-3600}"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

LOG="$LOG_DIR/qwen35_countdown_eggroll_D30_3h_metrics_disjoint_random_phased3x10_1h.log"
echo "=== EGGROLL D30 metrics phased 3x10 @ 1h (${TIME_BUDGET}s) -> $LOG ==="

"$VENV_PYTHON" -u -m llm_experiments.general_do_evolution \
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
  --eval-train-every 5 \
  --train-dataset-size 30 \
  --val-dataset-size 256 \
  --time-budget-seconds "$TIME_BUDGET" \
  --random-train-prompts \
  --train-phased-subset-sizes "10,10,10" \
  --train-phased-durations-seconds "${PHASE_SECONDS},${PHASE_SECONDS},${PHASE_SECONDS}" \
  --train-phased-split-seed 0 \
  --wandb-name "D30_3h_metrics_phased3x10_1h_disjoint_rand" \
  2>&1 | tee "$LOG"

echo "=== Finished D30 metrics phased 3x10 @ 1h (exit $?) ==="
