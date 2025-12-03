#!/bin/bash


CHECKPOINT="/orcd/pool/008/lerchen/trdm/TinyRecursiveModels/checkpoints/trm_sudoku_att_step21700.pt"
ONESTEP_CHECKPOINT="/home/rohin/trdm/save/core_all_one_step_final.pt"
DATA_DIR="/home/rohin/trdm/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000"
OUTPUT_DIR="analysis/rl_main_run"

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

cd /home/rohin/trdm/TinyRecursiveModels
source /home/rohin/trdm/.venv/bin/activate

torchrun \
    --standalone \
    --nproc_per_node=${SLURM_GPUS_PER_NODE:-1} \
    rl.py \
    --checkpoint "${CHECKPOINT}" \
    --onestep-checkpoint "${ONESTEP_CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --g 16 \
    --lr 1e-5 \
    --max-steps 100000 \
    --log-interval 10 \
    --ckpt-interval 500 \
    --eval-interval 1000 \
    --eval-puzzles 2048 \
    --noise-scale 0.02 \
    --steps 4 \
    --post-process-noise-scale 0.5
