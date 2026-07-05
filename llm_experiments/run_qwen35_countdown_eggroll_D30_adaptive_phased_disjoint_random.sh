#!/usr/bin/env bash
set -euo pipefail

# D30 adaptive phased 3x10: switch when pool>=90% and val stagnates/drops.
# Max phase caps 2400+1800+1800s; total budget 1.5h; early stop on global val decline.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
TIME_BUDGET="${TIME_BUDGET_SECONDS:-5400}"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

LOG="$LOG_DIR/qwen35_countdown_eggroll_D30_adaptive_phased_disjoint_random.log"
echo "=== EGGROLL D30 adaptive phased 3x10 (${TIME_BUDGET}s) -> $LOG ==="

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
  --train-phased-durations-seconds "2400,1800,1800" \
  --train-phased-split-seed 0 \
  --train-phased-adaptive \
  --train-phased-adaptive-pool-threshold 0.90 \
  --train-phased-adaptive-val-drop 0.03 \
  --train-phased-adaptive-early-stop-val-drop 0.03 \
  --train-phased-adaptive-min-seconds 600 \
  --wandb-name "D30_adaptive_phased3x10_disjoint_rand" \
  2>&1 | tee "$LOG"

echo "=== Finished D30 adaptive phased (exit $?) ==="
