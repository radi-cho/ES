#!/usr/bin/env bash
set -euo pipefail

# Monitor pop256 EGGROLL+GRPO block: kill/restart on OOM, errors, or val <= 0.015.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
LOG="$REPO_ROOT/runs/qwen35_eggroll_grpo_2h_validation_watch.log"
MASTER="$REPO_ROOT/runs/qwen35_eggroll_grpo_2h_disjoint_random_pop256_resilient_master.log"
POLL="${POLL_SECONDS:-60}"

exec >> "$LOG" 2>&1
echo "=== Validation watch started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

is_oom_or_resource_error() {
  grep -qiE 'OutOfMemory|OOM|RESOURCE_EXHAUSTED|out of memory|CUDA error.*memory|Failed to allocate' "$1"
}

latest_run_dir() {
  local pattern="$1"
  ls -td "$REPO_ROOT"/${pattern} 2>/dev/null | head -1 || true
}

check_latest_validation() {
  local dir="$1"
  [[ -n "$dir" && -f "$dir/validation.csv" ]] || return 0
  if ! "$VENV_PYTHON" -m llm_experiments.check_validation_health "$dir/validation.csv"; then
    return 1
  fi
  return 0
}

kill_training() {
  echo "Killing training processes..."
  pkill -f "run_qwen35_countdown_eggroll_grpo_2h_disjoint_random_resilient" || true
  pkill -f "llm_experiments.general_do_evolution.*D256_2h_disjoint_rand" || true
  pkill -f "llm_experiments.general_do_evolution.*D8_2h_disjoint_rand" || true
  pkill -f "llm_experiments.do_grpo.*D256_2h_disjoint_rand" || true
  pkill -f "llm_experiments.do_grpo.*D8_2h_disjoint_rand" || true
  sleep 5
}

restart_master() {
  local reason="$1"
  echo "=== RESTART: $reason ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
  kill_training
  nohup bash "$REPO_ROOT/llm_experiments/run_qwen35_countdown_eggroll_grpo_2h_disjoint_random_resilient.sh" \
    >> "$REPO_ROOT/runs/qwen35_eggroll_grpo_2h_pop256_resilient_nohup.log" 2>&1 &
  echo "Restarted master PID $!"
}

while true; do
  if [[ -f "$MASTER" ]] && grep -q "All runs complete" "$MASTER"; then
    echo "All runs complete — plotting validation-only figure."
    bash "$REPO_ROOT/llm_experiments/plot_eggroll_grpo_2h_validation.sh" 8 2 || true
    echo "=== Watch done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    exit 0
  fi

  for pattern in \
    "countdown_chat_eggroll_D256_2h_disjoint_rand_*trainD=256*" \
    "countdown_chat_eggroll_D8_2h_disjoint_rand_*trainD=8*" \
    "countdown_chat_grpo_D256_2h_disjoint_rand_*trainD=256*" \
    "countdown_chat_grpo_D8_2h_disjoint_rand_*trainD=8*"; do
    dir="$(latest_run_dir "$pattern")"
    if [[ -n "$dir" ]] && ! check_latest_validation "$dir"; then
      restart_master "bad validation in $(basename "$dir")"
      sleep "$POLL"
      continue 2
    fi
  done

  for runlog in "$REPO_ROOT"/runs/qwen35_countdown_eggroll_D*_2h_disjoint_random_bs*.log \
                "$REPO_ROOT"/runs/qwen35_countdown_grpo_D*_2h_disjoint_random_bs*.log; do
    [[ -f "$runlog" ]] || continue
    if is_oom_or_resource_error "$runlog"; then
      restart_master "OOM in $(basename "$runlog")"
      sleep "$POLL"
      continue 2
    fi
    if grep -qE 'Traceback \(most recent call last\)|ValueError:|RuntimeError:' "$runlog"; then
      if ! grep -q "Finished EGGROLL\|Finished GRPO\|All runs complete" "$runlog"; then
        restart_master "error in $(basename "$runlog")"
        sleep "$POLL"
        continue 2
      fi
    fi
  done

  sleep "$POLL"
done
