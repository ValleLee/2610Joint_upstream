#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if bash script/train_rgbdepth_k24_r100_perimg.sh; then
  run_dir=""
  if [[ -n "${RESUME_RUN:-}" ]]; then
    run_dir="$(cd "$(dirname "$RESUME_RUN")/.." && pwd)"
  else
    while IFS= read -r candidate; do
      [[ -f "$candidate/config.yaml" ]] || continue
      if grep -Eq '^[[:space:]]*main_val_metric:[[:space:]]*joint_latent_mse([[:space:]]|$)' \
        "$candidate/config.yaml"; then
        run_dir="$candidate"
        break
      fi
    done < <(
      find "$PROJECT_ROOT/output" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr | cut -d' ' -f2-
    )
  fi

  case "$run_dir" in
    "$PROJECT_ROOT"/output/*)
      if [[ -d "$run_dir" && -f "$run_dir/config.yaml" ]]; then
        touch "$run_dir/.training_complete"
        echo "[$(date '+%F %T')] marked successful training completion: $run_dir"
      else
        echo "[$(date '+%F %T')] training exited successfully but run directory was not found" >&2
        exit 2
      fi
      ;;
    *)
      echo "[$(date '+%F %T')] refusing to mark unexpected run directory: $run_dir" >&2
      exit 2
      ;;
  esac
else
  status=$?
  echo "[$(date '+%F %T')] training process exited with status $status; leaving run resumable" >&2
  exit "$status"
fi
