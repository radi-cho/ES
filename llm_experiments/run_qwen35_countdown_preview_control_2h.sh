#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
TRAIN_D="${TRAIN_D:-8}"
SEED="${SEED:-0}"
# Leave room for the existing preflight and compilation while retaining the
# same validation cadence as the 2-hour PR4/PR5 comparison.
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-${TIME_BUDGET:-6300}}"
NUM_EPOCHS="${NUM_EPOCHS:-100000}"
PHYSICAL_POPULATION="${PHYSICAL_POPULATION:-64}"
PHYSICAL_PER_PROMPT="${PHYSICAL_PER_PROMPT:-8}"
VIRTUAL_FACTOR="${VIRTUAL_FACTOR:-16}"
PREDICTIVE_USE_PREDICTIONS="${PREDICTIVE_USE_PREDICTIONS:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/qwen35_countdown_preview_control_D${TRAIN_D}_seed${SEED}_${RUN_TAG}}"
MASTER_LOG="${MASTER_LOG:-$REPO_ROOT/runs/qwen35_preview_control_D${TRAIN_D}_2h_master.log}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python environment not found at $VENV_PYTHON; set VENV_PYTHON explicitly." >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/.mplconfig}"

mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$(dirname "$MASTER_LOG")"
cd "$REPO_ROOT"

VIRTUAL_POPULATION=$((PHYSICAL_POPULATION * VIRTUAL_FACTOR))
AUDIT_PAIRS=$((PHYSICAL_POPULATION / 2))
VIRTUAL_PAIRS=$((VIRTUAL_POPULATION / 2))
echo "PR6 preview-control EGGROLL"
echo "Physical population: $PHYSICAL_POPULATION ($PHYSICAL_PER_PROMPT per prompt)"
echo "Virtual population: $VIRTUAL_POPULATION ($VIRTUAL_PAIRS pairs)"
echo "Full-rollout audits: $AUDIT_PAIRS pairs (${PHYSICAL_POPULATION} trajectories)"
echo "Feature: centered sketch_summary; surrogate: calibrated online ridge"
echo "Training budget: $TIME_BUDGET_SECONDS seconds"

PREDICTION_FLAGS=()
if [[ "$PREDICTIVE_USE_PREDICTIONS" != "1" ]]; then
  PREDICTION_FLAGS+=(--no-predictive-use-predictions)
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
  --noiser predictive_eggroll \
  --model-choice q35_2B \
  --rwkv-type Qwen35RWKV \
  --parallel-generations-per-gpu "$PHYSICAL_POPULATION" \
  --generations-per-prompt "$PHYSICAL_PER_PROMPT" \
  --predictive-virtual-factor "$VIRTUAL_FACTOR" \
  --predictive-preview-microbatch-pairs "${PREVIEW_MICROBATCH_PAIRS:-8}" \
  --predictive-feature-kind sketch_summary \
  --predictive-prompt-center \
  --predictive-surrogate ridge \
  --predictive-sketch-size "${PREDICTIVE_SKETCH_SIZE:-128}" \
  --predictive-ridge "${PREDICTIVE_RIDGE:-10.0}" \
  --predictive-decay "${PREDICTIVE_DECAY:-0.99}" \
  --predictive-min-observations "${PREDICTIVE_MIN_OBSERVATIONS:-128}" \
  --predictive-prediction-clip "${PREDICTIVE_PREDICTION_CLIP:-1.1}" \
  --predictive-reward-scale "${PREDICTIVE_REWARD_SCALE:-0.5}" \
  --predictive-reward-scale-decay "${PREDICTIVE_REWARD_SCALE_DECAY:-0.9}" \
  --predictive-minimum-reward-scale "${PREDICTIVE_MINIMUM_REWARD_SCALE:-0.1}" \
  --predictive-feature-seed "${PREDICTIVE_FEATURE_SEED:-0}" \
  --predictive-audit-seed "${PREDICTIVE_AUDIT_SEED:-1}" \
  --predictive-rms-floor "${PREDICTIVE_RMS_FLOOR:-1e-4}" \
  --predictive-calibration-decay "${PREDICTIVE_CALIBRATION_DECAY:-0.9}" \
  --predictive-calibration-max-scale "${PREDICTIVE_CALIBRATION_MAX_SCALE:-1.0}" \
  --predictive-calibration-min-observations "${PREDICTIVE_CALIBRATION_MIN_OBSERVATIONS:-32}" \
  --predictive-calibration-prior-observations "${PREDICTIVE_CALIBRATION_PRIOR_OBSERVATIONS:-64}" \
  "${PREDICTION_FLAGS[@]}" \
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
  --time-budget-seconds "$TIME_BUDGET_SECONDS" \
  --random-train-prompts \
  --output-directory "$RUN_ROOT" \
  --wandb-name "preview_control_D${TRAIN_D}_seed${SEED}_2h" \
  2>&1 | tee "$MASTER_LOG"

echo "Completed: $RUN_ROOT"
