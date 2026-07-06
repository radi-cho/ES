#!/usr/bin/env bash
set -euo pipefail

# Run on the same host/environment as the Figure-4b baselines:
#   CUDA_VISIBLE_DEVICES=0 TRAIN_D=256 TIME_BUDGET_SECONDS=7200 \
#     bash llm_experiments/run_qwen35_countdown_eggroll_vs_diag_2h.sh
# Repeat with TRAIN_D=8 for the low-data comparison.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
TIME_BUDGET="${TIME_BUDGET_SECONDS:-7200}"
NUM_EPOCHS="${NUM_EPOCHS:-1000000}"
TRAIN_D="${TRAIN_D:-256}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
AB_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/eggroll_vs_diag_D${TRAIN_D}_${RUN_TAG}}"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$AB_ROOT/.mplconfig}"

if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Expose exactly one GPU (for example CUDA_VISIBLE_DEVICES=0)." >&2
  exit 2
fi

mkdir -p "$AB_ROOT" "$MPLCONFIGDIR"
cd "$REPO_ROOT"

"$VENV_PYTHON" - <<'PY'
import jax

devices = jax.devices()
if len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit(f"Expected exactly one JAX GPU, found: {devices}")
print(f"Using JAX device: {devices[0]}")
PY

if [[ "${RUN_PREFLIGHT:-1}" == "1" ]]; then
  PREFLIGHT_LOG="$AB_ROOT/epoch0_preflight.log"
  "$VENV_PYTHON" -m llm_experiments.eval_countdown_chat_epoch0 \
    2>&1 | tee "$PREFLIGHT_LOG"
  awk '
    /EPOCH 0 VALIDATION SCORE/ { found=1; if ($NF + 0 < 0.05) exit 1 }
    END { if (!found) exit 2 }
  ' "$PREFLIGHT_LOG" || {
    echo "Epoch-0 score was missing or below 0.05; refusing to start training." >&2
    exit 1
  }
fi

run_case() {
  local method="$1"
  local case_dir="$AB_ROOT/$method"
  local log="$AB_ROOT/$method.log"

  mkdir -p "$case_dir"
  echo "=== $method, D=$TRAIN_D, budget=${TIME_BUDGET}s ==="
  "$VENV_PYTHON" -m llm_experiments.general_do_evolution \
    --task countdown_chat \
    --noiser "$method" \
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
    --num-epochs "$NUM_EPOCHS" \
    --validate-every 5 \
    --train-dataset-size "$TRAIN_D" \
    --val-dataset-size 256 \
    --time-budget-seconds "$TIME_BUDGET" \
    --random-train-prompts \
    --output-directory "$case_dir" \
    --wandb-name "D${TRAIN_D}_2h_disjoint_rand" \
    2>&1 | tee "$log"
}

run_case eggroll
run_case diag_eggroll

BASE_CSV="$(find "$AB_ROOT/eggroll" -name validation.csv -type f -print -quit)"
DIAG_CSV="$(find "$AB_ROOT/diag_eggroll" -name validation.csv -type f -print -quit)"
if [[ -z "$BASE_CSV" || -z "$DIAG_CSV" ]]; then
  echo "Missing validation.csv for one or both methods under $AB_ROOT" >&2
  exit 1
fi

"$VENV_PYTHON" -m llm_experiments.plot_figure_4b \
  "$BASE_CSV" \
  "$DIAG_CSV" \
  --label "EGGROLL train D=$TRAIN_D" \
  --label "Diag-Kron EGGROLL train D=$TRAIN_D" \
  --max-hours 2 \
  --title "Countdown validation — Qwen3.5-2B (EGGROLL vs Diag-Kron, D=$TRAIN_D)" \
  --output "$AB_ROOT/eggroll_vs_diag_D${TRAIN_D}_2h.png"

echo "Completed: $AB_ROOT"
