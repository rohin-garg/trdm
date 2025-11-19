#!/bin/bash
#SBATCH -p sched_mit_kburdge_r8          # your group partition
#SBATCH --job-name=sudoku_flow           # name for the job
#SBATCH --gres=gpu:4                     # 4 GPUs (matches CUDA_VISIBLE_DEVICES=0,1,2,3)
#SBATCH --ntasks=1                       # one main process (torchrun handles ranks)
#SBATCH --cpus-per-task=32               # number of CPU cores
#SBATCH --mem=128G                       # RAM (tune as needed)
#SBATCH --time=2-00:00:00                # up to 2 days (format: D-HH:MM:SS)
#SBATCH --output=slurm-%j.out            # Slurm log; your script also logs to $LOG

# If you need modules for CUDA, Python, etc., load them here, e.g.:
# module load miniforge
# module load cuda/12.0

# Run your training script
bash TinyRecursiveModels/run_flow_matching_916_layer.sh
