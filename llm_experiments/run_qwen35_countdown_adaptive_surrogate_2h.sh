#!/usr/bin/env bash
set -euo pipefail

# Adaptive surrogate-fitness EGGROLL:
# epoch X score -> epoch X+1 surrogate fraction.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
TIME_BUDGET="${TIME_BUDGET_SECONDS:-7200}"
NUM_EPOCHS="${NUM_EPOCHS:-1000000}"
SEED="${SEED:-0}"
TRAIN_D="${TRAIN_D:-8}"
PHYSICAL_POPULATION="${PHYSICAL_POPULATION:-64}"
PHYSICAL_PER_PROMPT="${PHYSICAL_PER_PROMPT:-8}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/qwen35_countdown_adaptive_surrogate_D${TRAIN_D}_seed${SEED}_${RUN_TAG}}"
MASTER_LOG="${MASTER_LOG:-$REPO_ROOT/runs/qwen35_adaptive_surrogate_D${TRAIN_D}_pop64_2h_master.log}"

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

echo "Adaptive surrogate EGGROLL"
echo "Population: $PHYSICAL_POPULATION ($PHYSICAL_PER_PROMPT per prompt)"
echo "Trust score -> next-epoch surrogate fraction"
echo "Max surrogate fraction: ${SURROGATE_MAX_FRACTION:-0.8}"

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
  --noiser adaptive_surrogate_eggroll \
  --model-choice q35_2B \
  --rwkv-type Qwen35RWKV \
  --parallel-generations-per-gpu "$PHYSICAL_POPULATION" \
  --generations-per-prompt "$PHYSICAL_PER_PROMPT" \
  --surrogate-probes-per-matrix "${SURROGATE_PROBES_PER_MATRIX:-2}" \
  --surrogate-ridge "${SURROGATE_RIDGE:-16.0}" \
  --surrogate-replay-capacity "${SURROGATE_REPLAY_CAPACITY:-64}" \
  --surrogate-min-observations "${SURROGATE_MIN_OBSERVATIONS:-4}" \
  --surrogate-feature-seed "${SURROGATE_FEATURE_SEED:-0}" \
  --surrogate-max-fraction "${SURROGATE_MAX_FRACTION:-0.8}" \
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
  --wandb-name "adaptive_surrogate_D${TRAIN_D}_seed${SEED}_2h_disjoint_rand" \
  2>&1 | tee "$MASTER_LOG"

echo "Completed: $RUN_ROOT"
