#!/usr/bin/env bash
set -euo pipefail

# Master: EGGROLL D=30, 3h each — full pool vs phased 3x10 (~6h total on 1 GPU).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/runs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/qwen35_countdown_eggroll_D30_3h_master.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=== Master: EGGROLL D30 3h disjoint+random (full vs phased 3x10) ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

bash "$REPO_ROOT/llm_experiments/run_qwen35_countdown_eggroll_D30_3h_disjoint_random.sh"
bash "$REPO_ROOT/llm_experiments/run_qwen35_countdown_eggroll_D30_3h_phased_disjoint_random.sh"

echo "=== All D30 3h runs complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
