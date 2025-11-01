#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(realpath "${SCRIPT_DIR}/TinyRecursiveModels")"
VENV="$(realpath "${SCRIPT_DIR}/.venv")"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$ROOT/outputs/diffusion_916_streaming_arcagi2_${TIMESTAMP}"
LOG="$OUT_DIR/stdout.log"

mkdir -p "$OUT_DIR"
: > "$LOG"

source "$VENV/bin/activate"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=14400
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nproc_per_node=4 --master_port=29931 \
  train_diffusion_916_streaming.py \
  --checkpoint-path "$ROOT/checkpoints/arc1concept/world_arcagi2_from_world_fresh/step_1452180" \
  --data-dir "$ROOT/data/arc2concept-aug-1000" \
  --output-dir "$OUT_DIR" \
  --trm-global-batch-size 32 \
  --train-batch-size 64 \
  --num-epochs 50 \
  --time-limit-seconds 86400 \
  --log-interval 50 \
  --grad-accumulation 1 \
  --dtype float32 \
  --max-trm-batches-per-epoch 32000 \
  --eval-interval-steps 3000 \
  --eval-num-samples 1 \
  --eval-num-aug-samples 8 \
  --eval-trm-batch-size 16 \
  --eval-pass-ks 1 2 \
  --eval-submission-k 2 \
  --eval-device cuda:0 \
  --eval-after-train >> "$LOG" 2>&1
