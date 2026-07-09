#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="/home/siana/HyperscaleES_v2_308c579/.venv/bin/python"
VENV_PYTHON="${VENV_PYTHON:-$DEFAULT_PYTHON}"
OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-$REPO_ROOT/outputs/countdown_oracle_q35_2b_D256_P32_seed0}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Python environment not found at $VENV_PYTHON; set VENV_PYTHON explicitly." >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$REPO_ROOT"
exec "$VENV_PYTHON" -m llm_experiments.collect_countdown_oracle_dataset \
  --output-directory "$OUTPUT_DIRECTORY" \
  --dataset-size "${DATASET_SIZE:-256}" \
  --directions-per-prompt "${DIRECTIONS_PER_PROMPT:-32}" \
  --generation-length "${GENERATION_LENGTH:-1024}" \
  --seed "${SEED:-0}" \
  --sigma "${SIGMA:-1e-3}" \
  --center-rms-floor "${CENTER_RMS_FLOOR:-1e-4}" \
  --val-holdout-size "${VAL_HOLDOUT_SIZE:-256}"
