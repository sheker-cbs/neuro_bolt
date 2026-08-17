#!/bin/bash
#SBATCH --job-name=neurobolt
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs_baseline/output_logs/neurobolt_%j.out
#SBATCH --error=logs_baseline/error_logs/neurobolt_%j.err
#
# =============================================================================
# NeuroBOLT SLURM job (SHiVAi-style): one batch per allocation, no --array.
# Everything runs inside this job (no nested sbatch).
# =============================================================================
#
# Prerequisites:
#   - conda env `neurobolt` available after `source ~/.bashrc`
#   - sbatch FROM this code root (relative #SBATCH log paths)
#   - mkdir -p logs_baseline/output_logs logs_baseline/error_logs
#     (script also creates these; attention mode uses logs_attn/)
#   - Optional: bash prepare_subject_batches.sh  → subject_batches/batch_NNN.txt
#
# Usage (preferred — job name set on the CLI; #SBATCH --job-name does not
# expand bash variables at submit time):
#   sbatch --job-name=neurobolt_batch_001 run_neuro.sh subject_batches/batch_001.txt
#   sbatch --job-name=neurobolt_batch_001 run_neuro.sh batch_001
#   bash submit_all_subjects.sh
#
# Env-only (no $1) — same as before:
#   NEUROBOLT_SUBJECT=sub-001 sbatch --export=ALL run_neuro.sh
#   NEUROBOLT_SUBJECTS=sub-001,sub-002 sbatch --export=ALL run_neuro.sh
#   (unset both) → ALL subjects (~16 h; use OPTION A below)
#
# $1 = batch file path OR name under subject_batches/:
#   subject_batches/batch_001.txt | batch_001.txt | batch_001
#   → reads lines, exports NEUROBOLT_SUBJECTS=sub-a,sub-b,... , runs python
#
# GPUs: prefer ampere (valid → PENDING when busy).
#   #SBATCH --gres=gpu:ampere:1     ← live default
#   #SBATCH --gres=gpu:turing:1     ← if launch-fails, check sinfo GRES name
# Generic --gres=gpu:1 is often invalid here.
#
# Training mode (MUST match NEUROBOLT_TRAINING_MODE export below):
#   baseline  → logs_baseline/ + checkpoints/…/log_baseline/
#   attention → logs_attn/     + checkpoints/…/log_attn/
# For attention: change the two #SBATCH --output/--error lines above to
#   logs_attn/output_logs/... and logs_attn/error_logs/...
# then: NEUROBOLT_TRAINING_MODE=attention sbatch run_neuro.sh …
#
# Checkpoints / TB:
#   checkpoints/log_baseline/runs/{ts}_j{JOB}_baseline_{subject}/
#   checkpoints/log_attn/runs/{ts}_j{JOB}_attn_{subject}/
# TensorBoard:
#   tensorboard --logdir checkpoints/log_baseline/runs
#   tensorboard --logdir checkpoints/log_attn/runs
#
# =============================================================================
# OPTION A — longer partition, ALL subjects in one job (not live).
# Raise --time, drop NEUROBOLT_SUBJECT / NEUROBOLT_SUBJECTS / $1 so main.py
# loops every subject (~16–18 h).
# -----------------------------------------------------------------------------
# #SBATCH --job-name=neurobolt
# #SBATCH --gres=gpu:ampere:1
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

set -euo pipefail

source ~/.bashrc
conda activate neurobolt

CODE_ROOT="/data/p_03183/personal_workspaces/sheker/NeuroBOLT"
cd "$CODE_ROOT"

# Optional $1: batch file path or name under subject_batches/
BATCH_ARG="${1:-}"
if [ -n "$BATCH_ARG" ]; then
  resolve_batch_file() {
    local arg="$1"
    local candidates=()
    if [[ "$arg" = /* ]]; then
      candidates+=("$arg")
    else
      candidates+=("$CODE_ROOT/$arg" "$arg")
    fi
    candidates+=(
      "$CODE_ROOT/subject_batches/$arg"
      "$CODE_ROOT/subject_batches/${arg}.txt"
      "subject_batches/$arg"
      "subject_batches/${arg}.txt"
    )
    local c
    for c in "${candidates[@]}"; do
      if [ -f "$c" ]; then
        # Absolute path for reliable reads
        (cd "$(dirname "$c")" && echo "$(pwd)/$(basename "$c")")
        return 0
      fi
    done
    return 1
  }

  if ! BATCH_FILE="$(resolve_batch_file "$BATCH_ARG")"; then
    echo "ERROR: batch file not found for arg: $BATCH_ARG" >&2
    echo "  Expected e.g. subject_batches/batch_001.txt (under $CODE_ROOT)" >&2
    exit 1
  fi

  BATCH_SUBJECTS=()
  while IFS= read -r line; do
    [ -n "$line" ] && BATCH_SUBJECTS+=("$line")
  done < <(
    grep -E '^[[:space:]]*sub-' "$BATCH_FILE" | sed 's/[[:space:]]*$//' | grep -v '^$'
  )
  if [ "${#BATCH_SUBJECTS[@]}" -eq 0 ]; then
    echo "ERROR: no subject ids in batch file: $BATCH_FILE" >&2
    exit 1
  fi

  # Batch file wins over any inherited NEUROBOLT_SUBJECT (select_subjects
  # checks SUBJECT before SUBJECTS).
  unset NEUROBOLT_SUBJECT || true
  export NEUROBOLT_SUBJECT=""
  export NEUROBOLT_SUBJECTS
  NEUROBOLT_SUBJECTS=$(IFS=,; echo "${BATCH_SUBJECTS[*]}")
  echo "Batch file: $BATCH_FILE"
  echo "Loaded ${#BATCH_SUBJECTS[@]} subject(s) → NEUROBOLT_SUBJECTS=${NEUROBOLT_SUBJECTS}"
fi

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Flip to "attention" for the experimental stack (and #SBATCH paths above).
export NEUROBOLT_TRAINING_MODE="${NEUROBOLT_TRAINING_MODE:-baseline}"

if [ "$NEUROBOLT_TRAINING_MODE" = "attention" ] || [ "$NEUROBOLT_TRAINING_MODE" = "attn" ]; then
  mkdir -p logs_attn/output_logs logs_attn/error_logs
else
  mkdir -p logs_baseline/output_logs logs_baseline/error_logs
fi

echo "NEUROBOLT_TRAINING_MODE=${NEUROBOLT_TRAINING_MODE}"
echo "NEUROBOLT_SUBJECT=${NEUROBOLT_SUBJECT:-}"
echo "NEUROBOLT_SUBJECTS=${NEUROBOLT_SUBJECTS:-}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"

# Config lives in main.py (Namespace). Subjects from batch $1 /
# NEUROBOLT_SUBJECT / NEUROBOLT_SUBJECTS / optional SLURM_ARRAY_TASK_ID /
# or all if unset.
python main.py
