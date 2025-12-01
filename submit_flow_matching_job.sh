#!/bin/bash
#SBATCH -p mit_preemptable                
#SBATCH --job-name=sudoku_flow
#SBATCH --gres=gpu:4                     # request 4 GPUs (all H200s on the same node)
#SBATCH --constraint=h200                # *** H200 GPUs only ***
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=5:45:00                   # mit_normal_gpu max is 6 hours
#SBATCH --output=slurm-%j.out

# Load modules if needed:
# module load miniforge
# module load cuda/12.0

# Run your script
bash TinyRecursiveModels/run_flow_matching_916_layer.sh
