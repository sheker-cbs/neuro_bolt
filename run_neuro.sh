#!/bin/bash
#SBATCH --job-name=neurobolt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/output_logs/neurobolt_%A_%a.out
#SBATCH --error=logs/error_logs/neurobolt_%A_%a.err

source ~/.bashrc
conda activate neurobolt
cd /data/p_03183/personal_workspaces/sheker/NeuroBOLT

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python main.py




