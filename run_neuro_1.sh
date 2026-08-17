#!/bin/bash
#SBATCH --job-name=neurobolt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs_baseline/output_logs/neurobolt_%j.out
#SBATCH --error=logs_baseline/error_logs/neurobolt_%j.err


##SBATCH --job-name=neurobolt
##SBATCH --gres=gpu:turing:1
##SBATCH --cpus-per-task=8
##SBATCH --partition=short
##SBATCH --mem=24G
##SBATCH --time=3:00:00
##SBATCH --array=0-31
##SBATCH --output=/data/p_03183/personal_workspaces/sheker/NeuroBOLT/logs_baseline/output_logs/neurobolt_%j.out
##SBATCH --error=/data/p_03183/personal_workspaces/sheker/NeuroBOLT/logs_baseline/error_logs/neurobolt_%j.out


# ---------------------------------------------------------------------------
# Training mode (MUST match main.py / NEUROBOLT_TRAINING_MODE)
#   baseline  → SLURM logs under logs_baseline/ ; checkpoints under …/log_baseline/
#   attention → SLURM logs under logs_attn/     ; checkpoints under …/log_attn/
#
# #SBATCH --output/--error are fixed at submit time — flip BOTH the paths below
# AND the NEUROBOLT_TRAINING_MODE export when switching modes.
# ---------------------------------------------------------------------------
# --- LIVE DEFAULT: baseline ---
#SBATCH --output=logs_baseline/output_logs/neurobolt_%A_%a.out
#SBATCH --error=logs_baseline/error_logs/neurobolt_%A_%a.err
#
# --- Attention (comment baseline #SBATCH output/error above; uncomment these): ---
# #SBATCH --output=logs_attn/output_logs/neurobolt_%A_%a.out
# #SBATCH --error=logs_attn/error_logs/neurobolt_%A_%a.err
#
# LIVE DEFAULT: one Algermissen subject per array task (~30 min, fits `short`).
# Submit FROM this code root so relative log paths resolve:
#   sbatch run_neuro.sh
# Attention:
#   (flip #SBATCH paths above, then)
#   NEUROBOLT_TRAINING_MODE=attention sbatch run_neuro.sh
#   # or rely on the export below after editing it to attention
#
# Array mapping (0-based, after ALGERMISSEN_SKIP = sub-004, 015, 025):
#   SLURM_ARRAY_TASK_ID  ->  sorted subject list printed at the top of the .out
# 32 remaining subjects => --array=0-31. Change the range if the list changes.
# Override one task:  NEUROBOLT_SUBJECT=sub-001 sbatch --array=0 run_neuro.sh
#
# Checkpoints / TB:
#   checkpoints/log_baseline/runs/{ts}_j{JOB}_a{ARRAY}_baseline_{subject}/
#   checkpoints/log_attn/runs/{ts}_j{JOB}_a{ARRAY}_attn_{subject}/
# TensorBoard:
#   tensorboard --logdir checkpoints/log_baseline/runs
#   tensorboard --logdir checkpoints/log_attn/runs
#
# =============================================================================
# OPTION A — longer partition, ALL subjects in one job (not live).
# Copy-paste these #SBATCH lines INSTEAD of --array / %A_%a above, then
# submit without NEUROBOLT_SUBJECT so main.py loops every subject (~16–18 h).
# Use logs_baseline/ or logs_attn/ to match mode.
# -----------------------------------------------------------------------------
# #SBATCH --job-name=neurobolt
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=8
# #SBATCH --partition=gpu
# #SBATCH --mem=32G
# #SBATCH --time=20:00:00
# #SBATCH --output=logs_baseline/output_logs/neurobolt_%j.out
# #SBATCH --error=logs_baseline/error_logs/neurobolt_%j.err
# =============================================================================
#
# =============================================================================
# OPTION B — skip already-completed subjects (not live). See main.py loop.
# Check under checkpoints/log_baseline/runs/ or checkpoints/log_attn/runs/.

# =============================================================================

source ~/.bashrc
conda activate neurobolt
cd /data/p_03183/personal_workspaces/sheker/NeuroBOLT

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Flip to "attention" for the experimental stack (and #SBATCH paths above).
export NEUROBOLT_TRAINING_MODE="${NEUROBOLT_TRAINING_MODE:-baseline}"

if [ "$NEUROBOLT_TRAINING_MODE" = "attention" ] || [ "$NEUROBOLT_TRAINING_MODE" = "attn" ]; then
  mkdir -p logs_attn/output_logs logs_attn/error_logs
else
  mkdir -p logs_baseline/output_logs logs_baseline/error_logs
fi

echo "NEUROBOLT_TRAINING_MODE=${NEUROBOLT_TRAINING_MODE}"

# Config lives in main.py (Namespace). Subject from SLURM_ARRAY_TASK_ID
# or optional NEUROBOLT_SUBJECT; mode from NEUROBOLT_TRAINING_MODE.
python main.py
