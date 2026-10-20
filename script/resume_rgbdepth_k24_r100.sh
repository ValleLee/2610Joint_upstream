#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c "import sys; print('Python executable:', sys.executable)"

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output}"
export BASE_DATA_DIR="${BASE_DATA_DIR:-$PROJECT_ROOT}"
export BASE_CKPT_DIR="${BASE_CKPT_DIR:-$PROJECT_ROOT}"
export PAIRS_DIR="${PAIRS_DIR:-$PROJECT_ROOT/data/pairs/k24_ratio100_val2_ignore}"
export RESUME_RUN="${RESUME_RUN:-}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ -z "$RESUME_RUN" || ! -f "$RESUME_RUN/trainer.ckpt" ]]; then
  echo "Set RESUME_RUN to a valid local checkpoint/latest directory before resuming." >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$PROJECT_ROOT/logs"

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$PROJECT_ROOT/script/depth/train.py" \
  --resume_run "$RESUME_RUN" \
  --base_data_dir "$BASE_DATA_DIR" \
  --base_ckpt_dir "$BASE_CKPT_DIR" \
  --no_wandb \
  --skip_code_snapshot \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee -a "$PROJECT_ROOT/logs/resume_rgbdepth_k24_r100_${NPROC_PER_NODE}gpu_$(date +%F_%H%M%S).log"
