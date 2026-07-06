#!/usr/bin/env bash
set -euo pipefail

# Run one independent Qwen3.5-2B Countdown experiment. For example:
#   METHOD=product_space_eggroll TRAIN_D=256 \
#     bash llm_experiments/run_qwen35_countdown_single_2h.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"
METHOD="${METHOD:?Set METHOD to eggroll, diag_eggroll, or product_space_eggroll}"
TRAIN_D="${TRAIN_D:?Set TRAIN_D to 8 or 256}"
TIME_BUDGET="${TIME_BUDGET_SECONDS:-7200}"
NUM_EPOCHS="${NUM_EPOCHS:-1000000}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/qwen35_countdown_${METHOD}_D${TRAIN_D}_seed${SEED}_${RUN_TAG}}"

case "$METHOD" in
  eggroll|diag_eggroll|product_space_eggroll) ;;
  *) echo "METHOD must be eggroll, diag_eggroll, or product_space_eggroll" >&2; exit 2 ;;
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

METHOD_ARGS=(--noiser "$METHOD")
if [[ "$METHOD" == "product_space_eggroll" ]]; then
  METHOD_ARGS+=(
    --product-space-rank "${PRODUCT_SPACE_RANK:-8}"
    --product-space-scout-pairs "${PRODUCT_SPACE_SCOUT_PAIRS:-2}"
    --product-space-warmup-pairs "${PRODUCT_SPACE_WARMUP_PAIRS:-256}"
    --product-space-geometry-lr "${PRODUCT_SPACE_GEOMETRY_LR:-0.02}"
    --product-space-geometry-ema-decay "${PRODUCT_SPACE_GEOMETRY_EMA_DECAY:-0.9}"
    --product-space-geometry-update-every "${PRODUCT_SPACE_GEOMETRY_UPDATE_EVERY:-1}"
  )
  if [[ "${PRODUCT_SPACE_CONTROL_VARIATE:-1}" == "0" ]]; then
    METHOD_ARGS+=(--no-product-space-control-variate)
  fi
fi

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
  "${METHOD_ARGS[@]}" \
  --model-choice q35_2B \
  --rwkv-type Qwen35RWKV \
  --parallel-generations-per-gpu 64 \
  --generations-per-prompt 8 \
  --sigma 1e-3 \
  --lr-scale 0.2 \
  --seed "$SEED" \
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
  --wandb-name "${METHOD}_D${TRAIN_D}_seed${SEED}_2h_disjoint_rand" \
  2>&1 | tee "$RUN_ROOT/train.log"

echo "Completed: $RUN_ROOT"
