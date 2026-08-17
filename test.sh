#!/bin/bash
# Minimal SLURM smoke test — NO array, NO conda, NO python.
# Use this to see whether the cluster can start a job and write logs at all.
#
# 1) Edit ABS_LOG_DIR below to a directory that EXISTS on the cluster
#    and is visible from compute nodes (your project space, not a laptop path).
# 2) mkdir -p "$ABS_LOG_DIR"
# 3) sbatch test_slurm_smoke.sh
# 4) Check the .out file and: sacct -j <ID> --format=JobID,State,ExitCode,Reason -P
#
#SBATCH --job-name=nb_smoke
#SBATCH --partition=group_servers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --output=/data/p_03183/personal_workspaces/sheker/NeuroBOLT/logs_baseline/output_logs/smoke_%j.out
#SBATCH --error=/data/p_03183/personal_workspaces/sheker/NeuroBOLT/logs_baseline/error_logs/smoke_%j.err
#
# If ampere stays PENDING: only ~1 ampere on short — wait, or try turing:
#   #SBATCH --gres=gpu:turing:1
# (turing launch-failure → wrong GRES name or node/prolog issue; ask admin)
#
set -x
echo "START $(date)"
echo "host=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
pwd
which python || true
nvidia-smi || echo "nvidia-smi failed"
echo "END $(date)"
