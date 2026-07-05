#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

FULL_CSV="$(ls -td "$REPO_ROOT"/countdown_chat_eggroll_D30_3h_metrics_full_disjoint_rand_*/metrics.csv 2>/dev/null | head -1)"
PHASED_CSV="$(ls -td "$REPO_ROOT"/countdown_chat_eggroll_D30_3h_metrics_phased3x10_1h_disjoint_rand_*/metrics.csv 2>/dev/null | head -1)"

if [[ -z "$FULL_CSV" || -z "$PHASED_CSV" ]]; then
  echo "Missing metrics.csv for full and/or phased metrics runs." >&2
  exit 1
fi

OUT="$REPO_ROOT/runs/D30_metrics_full_vs_phased_1h.png"

"$VENV_PYTHON" -m llm_experiments.plot_D30_metrics_compare \
  "$FULL_CSV" "$PHASED_CSV" \
  --label "D30 full (8/30 per epoch)" \
  --label "D30 phased 3×10 (1 h/phase, 8/10)" \
  --title "D30 metrics — full vs phased 1h (EggRoll, 3h)" \
  -o "$OUT"

echo "Saved: $OUT"
