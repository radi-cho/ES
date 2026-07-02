#!/usr/bin/env bash
set -euo pipefail

# EGGROLL countdown with Qwen3.5-2B (JAX/HyperscaleES, single GPU).
# Uses countdown.json chat/XML format (same as eggroll-vllm). Expect epoch-0 val ~0.30 with thinking-length 1024.
#
# First-time setup:
#   cd /home/siana/HyperscaleES_v2_rwkv_qwen
#   pip install -e .
#
# Reuse the existing HyperscaleES venv (JAX/CUDA already installed):
#   bash llm_experiments/run_qwen35_countdown.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Drop stale JAX cache if es_map was built with an older qrwkv6 (causes ZeroDivisionError).
Q35_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hyperscalees_cache/q35_2B_bfloat16.model"
if [[ -f "$Q35_CACHE" ]]; then
  CACHE_AGE_DAYS=$(( ( $(date +%s) - $(stat -c %Y "$Q35_CACHE") ) / 86400 ))
  if (( CACHE_AGE_DAYS > 0 )); then
    echo "Removing stale q35_2B cache ($Q35_CACHE, ${CACHE_AGE_DAYS}d old)"
    rm -f "$Q35_CACHE"
  fi
fi

cd "$REPO_ROOT"

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
  --validate-every 5
