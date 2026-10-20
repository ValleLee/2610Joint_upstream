#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c "import sys; print('Python executable:', sys.executable)"

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT}"
export CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/config/train_rgbdepth_teacher_perimg.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output}"
export BASE_DATA_DIR="${BASE_DATA_DIR:-$PROJECT_ROOT}"
export BASE_CKPT_DIR="${BASE_CKPT_DIR:-$PROJECT_ROOT}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PAIRS_DIR="${PAIRS_DIR:-$PROJECT_ROOT/data/pairs/k24_ratio100_val2_ignore}"
export TRAIN_PAIRS_FILENAME="${TRAIN_PAIRS_FILENAME:-train_pairs.txt}"
export VAL_PAIRS_FILENAME="${VAL_PAIRS_FILENAME:-val_pairs.txt}"
export VIS_PAIRS_FILENAME="${VIS_PAIRS_FILENAME:-val_pairs.txt}"

export RESIZE_H="${RESIZE_H:-432}"
export RESIZE_W="${RESIZE_W:-768}"
export DEFAULT_PROC_RES="${DEFAULT_PROC_RES:-768}"

export COND_FROM_DEPTH="${COND_FROM_DEPTH:-true}"
export TEACHER_FROM_DEPTH="${TEACHER_FROM_DEPTH:-true}"
export COND_REPR="${COND_REPR:-depth}"
export TEACHER_REPR="${TEACHER_REPR:-depth}"

export VAL_TARGET_REPR="${VAL_TARGET_REPR:-depth}"
export VAL_CLIP_TO_DATASET_RANGE="${VAL_CLIP_TO_DATASET_RANGE:-false}"
export MAIN_VAL_METRIC="${MAIN_VAL_METRIC:-joint_latent_mse}"

export MIN_MAX_QUANTILE="${MIN_MAX_QUANTILE:-0.01}"
export EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
export MAX_TRAIN_BATCH_SIZE="${MAX_TRAIN_BATCH_SIZE:-4}"

export MAX_ITER="${MAX_ITER:-15000}"
export LR_TOTAL_ITER="${LR_TOTAL_ITER:-18000}"
export WARMUP_STEPS="${WARMUP_STEPS:-200}"
export VAL_DENOISE_STEPS="${VAL_DENOISE_STEPS:-4}"

export SAVE_PERIOD="${SAVE_PERIOD:-1000}"
export VAL_PERIOD="${VAL_PERIOD:-1000}"
export VIS_PERIOD="${VIS_PERIOD:-1000}"
export BACKUP_PERIOD="${BACKUP_PERIOD:-2000}"
export EARLY_VAL_PERIOD="${EARLY_VAL_PERIOD:-500}"
export EARLY_VIS_PERIOD="${EARLY_VIS_PERIOD:-500}"
export SCHEDULE_SWITCH_ITER="${SCHEDULE_SWITCH_ITER:-6000}"
export VIS_HEAD_SAMPLES="${VIS_HEAD_SAMPLES:-50}"
export VIS_TAIL_SAMPLES="${VIS_TAIL_SAMPLES:-50}"
export RESUME_RUN="${RESUME_RUN:-}"

export EARLY_STOP_ENABLED="${EARLY_STOP_ENABLED:-true}"
export EARLY_STOP_START_ITER="${EARLY_STOP_START_ITER:-10000}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.00005}"
export EARLY_STOP_GOAL="${EARLY_STOP_GOAL:-minimize}"

mkdir -p "$PROJECT_ROOT/logs"

TRAIN_ARGS=(
  --config "$CONFIG_PATH"
  --base_data_dir "$BASE_DATA_DIR"
  --base_ckpt_dir "$BASE_CKPT_DIR"
  --no_wandb
  --skip_code_snapshot
)

if [[ -n "$RESUME_RUN" ]]; then
  TRAIN_ARGS+=(--resume_run "$RESUME_RUN")
else
  TRAIN_ARGS+=(--output_dir "$OUTPUT_DIR" --add_datetime_prefix)
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$PROJECT_ROOT/script/depth/train.py" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "$PROJECT_ROOT/logs/train_rgbdepth_k24_r100_perimg_${NPROC_PER_NODE}gpu_$(date +%F_%H%M%S).log"
