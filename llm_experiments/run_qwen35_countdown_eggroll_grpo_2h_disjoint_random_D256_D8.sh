#!/usr/bin/env bash
set -euo pipefail

# Full block: EGGROLL D256+D8 (2h each), then GRPO D256+D8 (2h each) — ~8h total on 1 GPU.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/qwen35_eggroll_grpo_2h_disjoint_random_D256_D8_master.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=== Master: EGGROLL+GRPO 2h disjoint+random D256 vs D8 ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

bash "$REPO_ROOT/llm_experiments/run_qwen35_countdown_data_efficiency_2h_disjoint_random.sh"
bash "$REPO_ROOT/llm_experiments/run_qwen35_countdown_grpo_data_efficiency_2h_disjoint_random.sh"

echo "=== All runs complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$VENV_PYTHON" -m llm_experiments.plot_eggroll_vs_grpo_compare \
  --repo-root "$REPO_ROOT" \
  --small-d 8 \
  --budget-hours 2
