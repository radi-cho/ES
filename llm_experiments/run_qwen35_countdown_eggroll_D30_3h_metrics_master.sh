#!/usr/bin/env bash
set -euo pipefail

# Master: D30 full vs phased @ 1h/phase with extended metrics (~6h on 1 GPU).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/qwen35_countdown_eggroll_D30_3h_metrics_master.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=== Master: D30 metrics full vs phased 3x10 @ 1h ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

bash "$REPO_ROOT/llm_experiments/run_qwen35_countdown_eggroll_D30_3h_metrics_disjoint_random.sh"
bash "$REPO_ROOT/llm_experiments/run_qwen35_countdown_eggroll_D30_3h_metrics_phased_disjoint_random.sh"

bash "$REPO_ROOT/llm_experiments/plot_D30_metrics_compare.sh"

echo "=== All D30 metrics runs complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
