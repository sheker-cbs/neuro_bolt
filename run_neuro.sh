#!/bin/bash
#SBATCH --job-name=neurobolt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/output_logs/neurobolt_%j.out
#SBATCH --error=logs/error_logs/neurobolt_%j.err 

source ~/.bashrc
conda activate neurobolt
cd /data/p_03183/personal_workspaces/sheker/NeuroBOLT

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# python main.py \
#   --finetune ./checkpoints/labram-base.pth \
#   --labels_roi VS \
#   --dataset mydata \
#   --train_test_mode full_test \
#   --prepro_datapath ./my_data_seq2one.pkl


