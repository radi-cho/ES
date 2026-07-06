#!/usr/bin/env bash
set -euo pipefail

# EGGROLL+GRPO 2h D256/D8 with OOM retry: pop 256 -> lighter val -> pop 128.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/qwen35_eggroll_grpo_2h_disjoint_random_pop256_resilient_master.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

is_oom_or_resource_error() {
  local log="$1"
  grep -qiE 'OutOfMemory|OOM|RESOURCE_EXHAUSTED|out of memory|CUDA error.*memory|Failed to allocate' "$log"
}

run_with_retries() {
  local script="$1"
  local block_name="$2"
  local tmp_log
  tmp_log="$(mktemp)"

  try_run() {
    local cfg_name="$1"
    shift
    echo "--- Trying $block_name / $cfg_name ---"
    : > "$tmp_log"
    set +e
    env "$@" bash "$script" 2>&1 | tee "$tmp_log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" -eq 0 ]]; then
      echo "=== $block_name succeeded with $cfg_name ==="
      return 0
    fi
    if is_oom_or_resource_error "$tmp_log"; then
      echo "OOM/resource error in $block_name ($cfg_name), trying next config..."
    else
      echo "Non-OOM failure in $block_name ($cfg_name), exit $rc — trying next config..."
    fi
    return 1
  }

  try_run pop256_default \
    POPULATION_SIZE=256 PARALLEL_VALIDATIONS=64 VALIDATION_ITERATIONS=10 GRPO_MAX_SEQ_LEN=64 \
    && { rm -f "$tmp_log"; return 0; }

  try_run pop256_light_val \
    POPULATION_SIZE=256 PARALLEL_VALIDATIONS=32 VALIDATION_ITERATIONS=5 GRPO_MAX_SEQ_LEN=64 \
    && { rm -f "$tmp_log"; return 0; }

  try_run pop256_light_grpo \
    POPULATION_SIZE=256 PARALLEL_VALIDATIONS=32 VALIDATION_ITERATIONS=5 GRPO_MAX_SEQ_LEN=32 \
    && { rm -f "$tmp_log"; return 0; }

  try_run pop128_default \
    POPULATION_SIZE=128 PARALLEL_VALIDATIONS=64 VALIDATION_ITERATIONS=10 GRPO_MAX_SEQ_LEN=64 \
    && { rm -f "$tmp_log"; return 0; }

  try_run pop128_light_val \
    POPULATION_SIZE=128 PARALLEL_VALIDATIONS=32 VALIDATION_ITERATIONS=5 GRPO_MAX_SEQ_LEN=64 \
    && { rm -f "$tmp_log"; return 0; }

  rm -f "$tmp_log"
  echo "ERROR: $block_name failed after all retry configs" >&2
  return 1
}

echo "=== Master: EGGROLL+GRPO 2h disjoint+random D256 vs D8 (pop256 resilient) ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

run_with_retries \
  "$REPO_ROOT/llm_experiments/run_qwen35_countdown_data_efficiency_2h_disjoint_random.sh" \
  "EGGROLL"

run_with_retries \
  "$REPO_ROOT/llm_experiments/run_qwen35_countdown_grpo_data_efficiency_2h_disjoint_random.sh" \
  "GRPO"

echo "=== All runs complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
bash "$REPO_ROOT/llm_experiments/plot_eggroll_grpo_2h_validation.sh" 8 2
