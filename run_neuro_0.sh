#!/bin/bash
#SBATCH --job-name=neurobolt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/output_logs/neurobolt_%j.out
#SBATCH --error=slurm_logs/error_logs/neurobolt_%j.err

source ~/.bashrc
conda activate neurobolt
cd /data/p_03183/personal_workspaces/sheker/NeuroBOLT

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python main.py




