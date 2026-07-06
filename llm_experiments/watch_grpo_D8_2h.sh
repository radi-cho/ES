#!/usr/bin/env bash
set -euo pipefail

# Watch GRPO D8 pop256 run: restart on crash/OOM/bad val; plot when time budget completes.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-/home/siana/HyperscaleES_v2_308c579/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
LOG="$REPO_ROOT/runs/qwen35_grpo_D8_2h_watch.log"
RUN_LOG="$REPO_ROOT/runs/qwen35_countdown_grpo_D8_2h_disjoint_random_bs256.log"
PIDFILE="$REPO_ROOT/runs/qwen35_grpo_D8_2h_train.pid"
POLL="${POLL_SECONDS:-60}"
GRACE_SECONDS="${GRACE_SECONDS:-420}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"

exec >> "$LOG" 2>&1
echo "=== GRPO D8 watch started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

restarts=0
last_start_epoch=0

now_epoch() { date +%s; }

is_training_running() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  pgrep -f "llm_experiments.do_grpo.*train-dataset-size 8" >/dev/null 2>&1 \
    || pgrep -f "run_qwen35_countdown_grpo_D8_2h_disjoint_random_only" >/dev/null 2>&1
}

log_segment_since_last_start() {
  [[ -f "$RUN_LOG" ]] || return 0
  local start
  start=$(grep -n "^=== GRPO D8 disjoint" "$RUN_LOG" | tail -1 | cut -d: -f1)
  [[ -n "$start" ]] || { cat "$RUN_LOG"; return; }
  tail -n +"$start" "$RUN_LOG"
}

is_oom_or_error() {
  local segment
  segment="$(log_segment_since_last_start)"
  [[ -n "$segment" ]] || return 1
  if grep -qiE 'OutOfMemory|OOM|RESOURCE_EXHAUSTED|out of memory|CUDA error.*memory|Failed to allocate' <<< "$segment"; then
    return 0
  fi
  if grep -qE 'Traceback \(most recent call last\)|RuntimeError:' <<< "$segment"; then
    if ! grep -q "Time budget (7200.0s) reached after epoch" <<< "$segment"; then
      return 0
    fi
  fi
  if grep -q "Terminated" <<< "$segment" && ! is_training_running; then
    return 0
  fi
  return 1
}

latest_grpo_d8_dir() {
  ls -td "$REPO_ROOT"/countdown_chat_grpo_D8_2h_disjoint_rand_*bs=256* 2>/dev/null | head -1 || true
}

run_completed_ok() {
  local segment
  segment="$(log_segment_since_last_start)"
  grep -q "Time budget (7200.0s) reached after epoch" <<< "$segment"
}

within_grace() {
  local now elapsed
  now=$(now_epoch)
  elapsed=$((now - last_start_epoch))
  [[ "$last_start_epoch" -gt 0 && "$elapsed" -lt "$GRACE_SECONDS" ]]
}

start_run() {
  echo "=== Starting GRPO D8 run (restart #${restarts}) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  if [[ ! -f "$REPO_ROOT/llm_experiments/utils.py" ]]; then
    git -C "$REPO_ROOT" restore llm_experiments/utils.py llm_experiments/check_validation_health.py 2>/dev/null || true
  fi
  bash "$REPO_ROOT/llm_experiments/run_qwen35_countdown_grpo_D8_2h_disjoint_random_only.sh" \
    >> "$REPO_ROOT/runs/qwen35_countdown_grpo_D8_2h_disjoint_random_only_nohup.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  echo "Started wrapper PID $pid"
  last_start_epoch=$(now_epoch)
}

kill_stale() {
  if within_grace; then
    echo "Within ${GRACE_SECONDS}s grace after start — not killing"
    return 0
  fi
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Killing stale wrapper PID $pid"
      kill "$pid" 2>/dev/null || true
      sleep 3
    fi
  fi
  pkill -f "llm_experiments.do_grpo.*train-dataset-size 8" 2>/dev/null || true
  sleep 3
}

if ! is_training_running; then
  if run_completed_ok; then
    echo "Run already completed successfully."
    bash "$REPO_ROOT/llm_experiments/plot_eggroll_grpo_2h_validation.sh" 8 2 \
      -o "$REPO_ROOT/runs/figure_eggroll_vs_grpo_validation_D8_vs_D256_2h_pop256.png" || true
    echo "=== Watch done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    exit 0
  fi
  start_run
  restarts=$((restarts + 1))
fi

while true; do
  if run_completed_ok; then
    echo "GRPO D8 completed (time budget reached)."
    bash "$REPO_ROOT/llm_experiments/plot_eggroll_grpo_2h_validation.sh" 8 2 \
      -o "$REPO_ROOT/runs/figure_eggroll_vs_grpo_validation_D8_vs_D256_2h_pop256.png" || true
    echo "=== Watch done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    exit 0
  fi

  if ! is_training_running; then
    if within_grace; then
      echo "Waiting (grace ${GRACE_SECONDS}s, started ${last_start_epoch}) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      sleep "$POLL"
      continue
    fi
    echo "Training not running $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
      echo "ERROR: max restarts ($MAX_RESTARTS) reached" >&2
      exit 1
    fi
    if is_oom_or_error || true; then
      kill_stale
      restarts=$((restarts + 1))
      start_run
    fi
    sleep "$POLL"
    continue
  fi

  dir="$(latest_grpo_d8_dir)"
  if [[ -n "$dir" && -f "$dir/validation.csv" ]]; then
    if ! "$VENV_PYTHON" -m llm_experiments.check_validation_health "$dir/validation.csv" 2>/dev/null; then
      echo "Bad validation in $(basename "$dir") — restarting"
      kill_stale
      restarts=$((restarts + 1))
      start_run
      sleep "$POLL"
      continue
    fi
  fi

  sleep "$POLL"
done
