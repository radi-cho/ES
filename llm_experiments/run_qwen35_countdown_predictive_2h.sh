#!/usr/bin/env bash
set -euo pipefail

# One Qwen3.5-2B Countdown run with eight real trajectories and 128 virtual
# trajectories per prompt.  The physical accelerator batch remains 64:
# eight D=8 prompts x eight fully generated EGGROLL members.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv/bin/python}"
TIME_BUDGET="${TIME_BUDGET_SECONDS:-7200}"
NUM_EPOCHS="${NUM_EPOCHS:-1000000}"
SEED="${SEED:-0}"
TRAIN_D="${TRAIN_D:-8}"
PHYSICAL_POPULATION="${PHYSICAL_POPULATION:-64}"
PHYSICAL_PER_PROMPT="${PHYSICAL_PER_PROMPT:-8}"
VIRTUAL_FACTOR="${VIRTUAL_FACTOR:-16}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/qwen35_countdown_predictive_D${TRAIN_D}_seed${SEED}_${RUN_TAG}}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python environment not found at $VENV_PYTHON; set VENV_PYTHON explicitly." >&2
  exit 2
fi
if [[ "$TRAIN_D" != "8" ]]; then
  echo "This minimal comparison launcher expects TRAIN_D=8." >&2
  exit 2
fi
if (( PHYSICAL_POPULATION % PHYSICAL_PER_PROMPT != 0 )); then
  echo "PHYSICAL_POPULATION must be divisible by PHYSICAL_PER_PROMPT." >&2
  exit 2
fi
if (( PHYSICAL_PER_PROMPT % 2 != 0 )); then
  echo "PHYSICAL_PER_PROMPT must be even for antithetic pairs." >&2
  exit 2
fi

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

PREDICTION_ARGS=()
if [[ "${PREDICTIVE_USE_PREDICTIONS:-1}" == "0" ]]; then
  PREDICTION_ARGS+=(--no-predictive-use-predictions)
fi

echo "Physical trajectories/prompt: $PHYSICAL_PER_PROMPT"
echo "Virtual trajectories/prompt: $((PHYSICAL_PER_PROMPT * VIRTUAL_FACTOR))"
echo "Audit probability: 1/$VIRTUAL_FACTOR"
echo "Total physical trajectory batch: $PHYSICAL_POPULATION"
echo "Total virtual update population: $((PHYSICAL_POPULATION * VIRTUAL_FACTOR))"

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
  --noiser predictive_eggroll \
  --model-choice q35_2B \
  --rwkv-type Qwen35RWKV \
  --parallel-generations-per-gpu "$PHYSICAL_POPULATION" \
  --generations-per-prompt "$PHYSICAL_PER_PROMPT" \
  --predictive-virtual-factor "$VIRTUAL_FACTOR" \
  --predictive-probes-per-matrix "${PREDICTIVE_PROBES_PER_MATRIX:-2}" \
  --predictive-ridge "${PREDICTIVE_RIDGE:-32.0}" \
  --predictive-replay-capacity "${PREDICTIVE_REPLAY_CAPACITY:-64}" \
  --predictive-min-observations "${PREDICTIVE_MIN_OBSERVATIONS:-16}" \
  --predictive-feature-seed "${PREDICTIVE_FEATURE_SEED:-0}" \
  "${PREDICTION_ARGS[@]}" \
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
  --wandb-name "predictive_D${TRAIN_D}_seed${SEED}_2h" \
  2>&1 | tee "$RUN_ROOT/train.log"

echo "Completed: $RUN_ROOT"
