#!/usr/bin/env bash
set -euo pipefail

# Plot D30 full (existing 3h run, truncated to 1.5h) vs phased 30min/phase (1.5h run).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

FULL_CSV="${1:-}"
PHASED_CSV="${2:-}"

if [[ -z "$FULL_CSV" ]]; then
  FULL_CSV="$(ls -td "$REPO_ROOT"/countdown_chat_eggroll_D30_3h_disjoint_rand_full_*/validation.csv 2>/dev/null | head -1)"
fi
if [[ -z "$PHASED_CSV" ]]; then
  PHASED_CSV="$(ls -td "$REPO_ROOT"/countdown_chat_eggroll_D30_1.5h_disjoint_rand_phased3x10_30min_*/validation.csv 2>/dev/null | head -1)"
fi

if [[ ! -f "$FULL_CSV" ]]; then
  echo "Missing full run validation.csv: $FULL_CSV" >&2
  exit 1
fi
if [[ ! -f "$PHASED_CSV" ]]; then
  echo "Missing phased 30min run validation.csv: $PHASED_CSV" >&2
  exit 1
fi

OUT="$REPO_ROOT/runs/D30_full_vs_phased_1.5h_30min_validation.png"

"$VENV_PYTHON" -m llm_experiments.plot_figure_4b \
  "$FULL_CSV" \
  "$PHASED_CSV" \
  --label "D30 full (8/30 per epoch)" \
  --label "D30 phased 3×10 (30 min/phase, 8/10)" \
  --title "Countdown validation — D30 full vs phased 30min (EggRoll, 1.5h)" \
  --max-hours 1.5 \
  -o "$OUT"

echo "Saved: $OUT"
