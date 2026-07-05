#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

FULL_M="$(ls -td "$REPO_ROOT"/countdown_chat_eggroll_D30_3h_metrics_full_disjoint_rand_*/metrics.csv 2>/dev/null | head -1)"
PHASED_M="$(ls -td "$REPO_ROOT"/countdown_chat_eggroll_D30_3h_metrics_phased3x10_1h_disjoint_rand_*/metrics.csv 2>/dev/null | head -1)"
ADAPT_M="$(ls -td "$REPO_ROOT"/countdown_chat_eggroll_D30_adaptive_phased3x10_disjoint_rand_*/metrics.csv 2>/dev/null | head -1)"

FULL_V="${FULL_M%/metrics.csv}/validation.csv"
PHASED_V="${PHASED_M%/metrics.csv}/validation.csv"
ADAPT_V="${ADAPT_M%/metrics.csv}/validation.csv"

MAX_HOURS="${1:-}"

for f in "$FULL_M" "$PHASED_M" "$ADAPT_M" "$FULL_V" "$PHASED_V" "$ADAPT_V"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f" >&2
    exit 1
  fi
done

"$VENV_PYTHON" -m llm_experiments.plot_D30_metrics_compare \
  "$FULL_M" "$PHASED_M" "$ADAPT_M" \
  --label "D30 full (8/30)" \
  --label "D30 fixed phased 1h/phase" \
  --label "D30 adaptive phased (early stop)" \
  --title "D30 — full vs fixed vs adaptive phased (EggRoll)" \
  -o "$REPO_ROOT/runs/D30_metrics_full_vs_fixed_vs_adaptive.png"

MAX_ARGS=()
if [[ -n "$MAX_HOURS" ]]; then
  MAX_ARGS=(--max-hours "$MAX_HOURS")
  OUT_VAL="$REPO_ROOT/runs/D30_validation_full_vs_fixed_vs_adaptive_${MAX_HOURS}h.png"
else
  OUT_VAL="$REPO_ROOT/runs/D30_validation_full_vs_fixed_vs_adaptive.png"
fi

"$VENV_PYTHON" -m llm_experiments.plot_figure_4b \
  "$FULL_V" "$PHASED_V" "$ADAPT_V" \
  --label "D30 full (8/30)" \
  --label "D30 fixed phased 1h/phase" \
  --label "D30 adaptive phased (early stop)" \
  --title "Countdown validation — full vs fixed vs adaptive phased" \
  "${MAX_ARGS[@]}" \
  -o "$OUT_VAL"

echo "Saved: $REPO_ROOT/runs/D30_metrics_full_vs_fixed_vs_adaptive.png"
echo "Saved: $OUT_VAL"
