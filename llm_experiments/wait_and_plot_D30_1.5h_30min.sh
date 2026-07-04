#!/usr/bin/env bash
set -euo pipefail

# Wait for the phased 1.5h run PID, then refresh the comparison plot.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID="${1:?usage: wait_and_plot_D30_1.5h_30min.sh <pid>}"
LOG="$REPO_ROOT/runs/D30_full_vs_phased_1.5h_30min_plot_watcher.log"

{
  echo "=== Plot watcher started: $(date -u +%Y-%m-%dT%H:%M:%SZ) PID=$PID ==="
  while kill -0 "$PID" 2>/dev/null; do
    sleep 30
  done
  echo "=== Run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  bash "$REPO_ROOT/llm_experiments/plot_D30_full_vs_phased_1.5h_30min.sh"
  echo "=== Final plot refreshed: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} >>"$LOG" 2>&1
