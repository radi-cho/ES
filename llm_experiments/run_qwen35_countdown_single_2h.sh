#!/usr/bin/env bash
set -euo pipefail

# Run one independent Qwen3.5-2B Countdown experiment:
#   METHOD=diag_eggroll TRAIN_D=256 \
#     bash llm_experiments/run_qwen35_countdown_single_2h.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"
METHOD="${METHOD:?Set METHOD to eggroll or diag_eggroll}"
TRAIN_D="${TRAIN_D:?Set TRAIN_D to 8 or 256}"
TIME_BUDGET="${TIME_BUDGET_SECONDS:-7200}"
NUM_EPOCHS="${NUM_EPOCHS:-1000000}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/qwen35_countdown_${METHOD}_D${TRAIN_D}_${RUN_TAG}}"

case "$METHOD" in
  eggroll|diag_eggroll) ;;
  *) echo "METHOD must be eggroll or diag_eggroll" >&2; exit 2 ;;
esac

case "$TRAIN_D" in
  8|256) ;;
  *) echo "TRAIN_D must be 8 or 256" >&2; exit 2 ;;
esac

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/.mplconfig}"

if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Expose exactly one GPU (for example CUDA_VISIBLE_DEVICES=0)." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR"
cd "$REPO_ROOT"

"$VENV_PYTHON" - <<'PY'
import jax

devices = jax.devices()
if len(devices) != 1 or devices[0].platform != "gpu":
    raise SystemExit(f"Expected exactly one JAX GPU, found: {devices}")
print(f"Using JAX device: {devices[0]}")
PY

if [[ "${RUN_PREFLIGHT:-1}" == "1" ]]; then
  PREFLIGHT_LOG="$RUN_ROOT/epoch0_preflight.log"
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

"$VENV_PYTHON" -m llm_experiments.general_do_evolution \
  --task countdown_chat \
  --noiser "$METHOD" \
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
  --output-directory "$RUN_ROOT" \
  --wandb-name "${METHOD}_D${TRAIN_D}_2h_disjoint_rand" \
  2>&1 | tee "$RUN_ROOT/train.log"

echo "Completed: $RUN_ROOT"
