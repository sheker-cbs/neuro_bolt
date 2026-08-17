#!/bin/bash
# Submit Algermissen subjects the SHiVAi way: one sbatch per batch file,
# no --array, no nested sbatch. Each job runs subjects sequentially on one GPU.
#
# Flow:
#   1) prepare_subject_batches.sh  → subject_batches/batch_NNN.txt
#   2) this script                 → sbatch --job-name=… run_neuro.sh <batchfile>
#
# Usage:
#   bash submit_all_subjects.sh
#   bash submit_all_subjects.sh --batch-size 5
#   SUBJECTS_PER_JOB=5 bash submit_all_subjects.sh
#   bash submit_all_subjects.sh --one-per-job
#   bash submit_all_subjects.sh --dry-run
#   bash submit_all_subjects.sh --reuse-batches   # skip prepare if files exist
#   bash submit_all_subjects.sh sub-001 sub-007 sub-012
#   bash submit_all_subjects.sh --batch-size 3 sub-001 sub-002 sub-003 sub-005
#
# Attention mode (also switch #SBATCH log paths in run_neuro.sh to logs_attn/):
#   NEUROBOLT_TRAINING_MODE=attention bash submit_all_subjects.sh
#
# Manual single batch:
#   sbatch --job-name=neurobolt_batch_001 run_neuro.sh subject_batches/batch_001.txt
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BATCH_DIR="${BATCH_DIR:-$SCRIPT_DIR/subject_batches}"
SUBJECTS_PER_JOB="${SUBJECTS_PER_JOB:-5}"
DRY_RUN=0
REUSE_BATCHES=0
EXPLICIT_SUBJECTS=()

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --reuse-batches) REUSE_BATCHES=1; shift ;;
    --one-per-job) SUBJECTS_PER_JOB=1; shift ;;
    --batch-size)
      SUBJECTS_PER_JOB="${2:?--batch-size requires an integer}"
      shift 2
      ;;
    --batch-size=*)
      SUBJECTS_PER_JOB="${1#*=}"
      shift
      ;;
    sub-*)
      EXPLICIT_SUBJECTS+=("$1")
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage 1
      ;;
  esac
done

if ! [[ "$SUBJECTS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SUBJECTS_PER_JOB / --batch-size must be a positive integer (got: $SUBJECTS_PER_JOB)" >&2
  exit 1
fi

# Estimate: ~30 min/subject; warn if batch may exceed short 3h.
MAX_SAFE=5
if (( SUBJECTS_PER_JOB > MAX_SAFE )); then
  echo "WARNING: SUBJECTS_PER_JOB=${SUBJECTS_PER_JOB} may exceed short 3h walltime (~30 min/subject)." >&2
  echo "         Prefer --batch-size ${MAX_SAFE} or raise #SBATCH --time in run_neuro.sh." >&2
fi

need_prepare=1
if (( REUSE_BATCHES )) && compgen -G "$BATCH_DIR"/batch_*.txt > /dev/null; then
  prev_size=""
  if [ -f "$BATCH_DIR/.batch_size" ]; then
    prev_size="$(tr -d '[:space:]' < "$BATCH_DIR/.batch_size" || true)"
  fi
  if [ -n "$prev_size" ] && [ "$prev_size" = "$SUBJECTS_PER_JOB" ] && (( ${#EXPLICIT_SUBJECTS[@]} == 0 )); then
    need_prepare=0
    echo "Reusing existing batches under ${BATCH_DIR}/ (size=${prev_size})."
  else
    echo "Regenerating batches (reuse skipped: size/explicit mismatch or missing .batch_size)."
  fi
fi

if (( need_prepare )); then
  prepare_args=(--batch-size "$SUBJECTS_PER_JOB" --batch-dir "$BATCH_DIR")
  if (( ${#EXPLICIT_SUBJECTS[@]} > 0 )); then
    prepare_args+=("${EXPLICIT_SUBJECTS[@]}")
  fi
  bash "$SCRIPT_DIR/prepare_subject_batches.sh" "${prepare_args[@]}"
fi

BATCH_FILES=()
while IFS= read -r line; do
  [ -n "$line" ] && BATCH_FILES+=("$line")
done < <(ls -1 "$BATCH_DIR"/batch_*.txt 2>/dev/null | sort)
if [ "${#BATCH_FILES[@]}" -eq 0 ]; then
  echo "ERROR: no batch files in $BATCH_DIR (run prepare_subject_batches.sh)" >&2
  exit 1
fi

MODE="${NEUROBOLT_TRAINING_MODE:-baseline}"
n_jobs=${#BATCH_FILES[@]}

echo
echo "Mode:     ${MODE}"
echo "Batches:  ${n_jobs}  (subjects/job target=${SUBJECTS_PER_JOB})"
echo "Walltime: ~$(( SUBJECTS_PER_JOB * 30 )) min/job budget (assume ~30 min/subject; short=180 min)"
echo

mkdir -p logs_baseline/output_logs logs_baseline/error_logs
if [ "$MODE" = "attention" ] || [ "$MODE" = "attn" ]; then
  mkdir -p logs_attn/output_logs logs_attn/error_logs
fi

# Pass batch path as $1 to run_neuro.sh (SLURM forwards args after the script).
# Job name via CLI — #SBATCH --job-name=neurobolt_$VAR does not expand at submit.
job_idx=0
for batchfile in "${BATCH_FILES[@]}"; do
  job_idx=$((job_idx + 1))
  batch_base="$(basename "$batchfile" .txt)"   # e.g. batch_001
  # Prefer path relative to code root so the job finds it after cd.
  rel_batch="subject_batches/$(basename "$batchfile")"
  n_in_batch="$(grep -cE '^[[:space:]]*sub-' "$batchfile" || true)"
  subjects_preview="$(tr '\n' ',' < "$batchfile" | sed 's/,$//')"

  echo "[${job_idx}/${n_jobs}] ${batch_base}  (${n_in_batch}): ${subjects_preview}"
  if (( DRY_RUN )); then
    echo "  (dry-run) sbatch --job-name=neurobolt_${batch_base} --export=ALL run_neuro.sh ${rel_batch}"
  else
    NEUROBOLT_TRAINING_MODE="${MODE}" \
      sbatch \
        --job-name="neurobolt_${batch_base}" \
        --export=ALL \
        "$SCRIPT_DIR/run_neuro.sh" \
        "$rel_batch"
  fi
done

echo
if (( DRY_RUN )); then
  echo "Dry-run only — no jobs submitted."
else
  echo "Done. Check with: squeue -u \$USER"
fi
