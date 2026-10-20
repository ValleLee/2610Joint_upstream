#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SESSION="joint_teacher_wait"
TRAIN_SESSION="joint_teacher_train"
POLL_SECONDS="${POLL_SECONDS:-30}"
FREE_LIMIT_MIB="${FREE_LIMIT_MIB:-500}"
STABLE_POLLS="${STABLE_POLLS:-3}"
MAX_ITER_EXPECTED="${MAX_ITER_EXPECTED:-15000}"
MAIN_VAL_METRIC_EXPECTED="${MAIN_VAL_METRIC_EXPECTED:-joint_latent_mse}"

mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/logs"

free_gpu_list() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v limit="$FREE_LIMIT_MIB" '{
        gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2)
        if (($2 + 0) < limit) print $1
      }'
}

stable_free_gpu_list() {
  local previous="" current="" i
  local -a candidates selected
  for ((i = 0; i < STABLE_POLLS; i++)); do
    mapfile -t candidates < <(free_gpu_list)
    if (( ${#candidates[@]} < 4 )); then
      return 1
    fi
    selected=("${candidates[@]:0:4}")
    current="$(IFS=,; echo "${selected[*]}")"
    if [[ -n "$previous" && "$current" != "$previous" ]]; then
      return 1
    fi
    previous="$current"
    (( i + 1 < STABLE_POLLS )) && sleep "$POLL_SECONDS"
  done
  printf '%s\n' "$previous"
}

latest_run_dir() {
  local candidate
  while IFS= read -r candidate; do
    [[ -f "$candidate/config.yaml" ]] || continue
    if grep -Eq "^[[:space:]]*main_val_metric:[[:space:]]*${MAIN_VAL_METRIC_EXPECTED}([[:space:]]|$)" \
      "$candidate/config.yaml"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(
    find "$PROJECT_ROOT/output" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | cut -d' ' -f2-
  )
  return 1
}

latest_resume_checkpoint() {
  local run_dir="$1"
  local ckpt="$run_dir/checkpoint/latest"
  if [[ -f "$ckpt/trainer.ckpt" && -f "$ckpt/unet/config.json" && -f "$ckpt/scheduler/scheduler_config.json" ]]; then
    printf '%s\n' "$ckpt"
  fi
}

run_max_iter() {
  local run_dir="$1" n base max=0
  shopt -s nullglob
  for p in "$run_dir"/checkpoint/iter_*; do
    base="${p##*/}"
    n="${base#iter_}"
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    (( 10#$n > max )) && max=$((10#$n))
  done
  shopt -u nullglob
  printf '%d\n' "$max"
}

run_is_complete() {
  local run_dir="$1" max_iter
  [[ -f "$run_dir/.training_complete" ]] && return 0
  max_iter="$(run_max_iter "$run_dir")"
  (( max_iter >= MAX_ITER_EXPECTED ))
}

launch_train() {
  local gpu_list="$1" resume_path="${2-}"
  if [[ -n "$resume_path" ]]; then
    echo "[$(date '+%F %T')] resuming from $resume_path on GPUs $gpu_list"
    screen -dmS "$TRAIN_SESSION" bash -lc \
      "cd '$PROJECT_ROOT' && export CUDA_VISIBLE_DEVICES='$gpu_list' NPROC_PER_NODE=4 RESUME_RUN='$resume_path' && exec bash script/run_training_mark_success.sh"
  else
    echo "[$(date '+%F %T')] launching on GPUs $gpu_list"
    screen -dmS "$TRAIN_SESSION" bash -lc \
      "cd '$PROJECT_ROOT' && export CUDA_VISIBLE_DEVICES='$gpu_list' NPROC_PER_NODE=4 && unset RESUME_RUN && exec bash script/run_training_mark_success.sh"
  fi
  echo "[$(date '+%F %T')] launched screen session ${TRAIN_SESSION}"
}

while true; do
  if screen -list 2>/dev/null | grep -q "\.${TRAIN_SESSION}[[:space:]]"; then
    echo "[$(date '+%F %T')] training screen is alive: ${TRAIN_SESSION}"
    sleep "$POLL_SECONDS"
    continue
  fi

  run_dir="$(latest_run_dir || true)"
  if [[ -n "$run_dir" ]] && run_is_complete "$run_dir"; then
    echo "[$(date '+%F %T')] training complete: $run_dir (iter >= $MAX_ITER_EXPECTED)"
    exit 0
  fi

  resume_path=""
  if [[ -n "$run_dir" ]]; then
    resume_path="$(latest_resume_checkpoint "$run_dir" || true)"
  fi

  if gpu_list="$(stable_free_gpu_list)"; then
    launch_train "$gpu_list" "$resume_path"
    sleep "$POLL_SECONDS"
    continue
  fi

  mapfile -t free_gpus < <(free_gpu_list)

  if [[ -n "$resume_path" ]]; then
    echo "[$(date '+%F %T')] waiting to resume: free GPUs=${#free_gpus[@]}, need=4"
  else
    echo "[$(date '+%F %T')] waiting: free GPUs=${#free_gpus[@]}, need=4"
  fi
  sleep "$POLL_SECONDS"
done
