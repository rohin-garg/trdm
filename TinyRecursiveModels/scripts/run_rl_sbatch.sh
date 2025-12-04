#!/bin/bash
#SBATCH -J trm-rl
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

# Optional: make sure OpenMP threads match CPU allocation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

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
    --max-steps 10000 \
    --log-interval 10 \
    --ckpt-interval 500 \
    --eval-interval 200 \
    --eval-puzzles 2048 \
    --noise-scale 0.02 \
    --steps 4 \
    --post-process-noise-scale 0.5 \
    --num-fixed-batches 1 \
    --eval-first


# slurm-trm-rl-6861281.out => num-fixed-batches = 100
# slurm-trm-rl-6867232.out => num-fixed-batches = 1 -- to test if training on a single value at least makes training successes go up

