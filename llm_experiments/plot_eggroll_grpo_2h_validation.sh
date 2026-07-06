#!/usr/bin/env bash
set -euo pipefail

# Validation-only figure for 2h EGGROLL vs GRPO D8/D256 runs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SMALL_D="${1:-8}"
BUDGET_H="${2:-2}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

exec "$VENV_PYTHON" -m llm_experiments.plot_eggroll_vs_grpo_compare \
  --repo-root "$REPO_ROOT" \
  --small-d "$SMALL_D" \
  --budget-hours "$BUDGET_H" \
  -o "$REPO_ROOT/runs/figure_eggroll_vs_grpo_validation_D${SMALL_D}_vs_D256_${BUDGET_H}h.png" \
  "$@"
