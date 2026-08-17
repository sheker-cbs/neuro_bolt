#!/bin/bash
# Discover Algermissen subjects and write subject_batches/batch_NNN.txt
# (one subject id per line) for SHiVAi-style NeuroBOLT SLURM submits.
#
# Prerequisites:
#   - Run from this code root (or rely on SCRIPT_DIR)
#   - DATA_ROOT readable (cluster path), OR pass explicit sub-XXX args
#
# Usage:
#   bash prepare_subject_batches.sh
#   bash prepare_subject_batches.sh --batch-size 5
#   SUBJECTS_PER_JOB=5 bash prepare_subject_batches.sh
#   bash prepare_subject_batches.sh --batch-size 1          # one subject per file
#   bash prepare_subject_batches.sh sub-001 sub-007 sub-012
#
# Env:
#   ALGERMISSEN_DATA_ROOT / DATA_ROOT — override default data dir
#   SUBJECTS_PER_JOB                  — default batch size (5)
#   BATCH_DIR                         — output dir (default: ./subject_batches)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-${ALGERMISSEN_DATA_ROOT:-/data/p_03183/data/pav_algermissen/derived/py_imported/per_block/for_neurobolt}}"
# Same skip set as main.py ALGERMISSEN_SKIP
SKIP_REGEX='^(sub-004|sub-015|sub-025)$'

SUBJECTS_PER_JOB="${SUBJECTS_PER_JOB:-5}"
BATCH_DIR="${BATCH_DIR:-$SCRIPT_DIR/subject_batches}"
EXPLICIT_SUBJECTS=()

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --batch-size)
      SUBJECTS_PER_JOB="${2:?--batch-size requires an integer}"
      shift 2
      ;;
    --batch-size=*)
      SUBJECTS_PER_JOB="${1#*=}"
      shift
      ;;
    --batch-dir)
      BATCH_DIR="${2:?--batch-dir requires a path}"
      shift 2
      ;;
    --batch-dir=*)
      BATCH_DIR="${1#*=}"
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

if (( ${#EXPLICIT_SUBJECTS[@]} > 0 )); then
  SUBJECTS=("${EXPLICIT_SUBJECTS[@]}")
else
  if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Algermissen data root not found: $DATA_ROOT" >&2
    echo "Set ALGERMISSEN_DATA_ROOT or DATA_ROOT, or pass explicit sub-XXX args." >&2
    exit 1
  fi
  SUBJECTS=()
  while IFS= read -r line; do
    [ -n "$line" ] && SUBJECTS+=("$line")
  done < <(
    find "$DATA_ROOT" -maxdepth 1 -name '*_block*.npz' -printf '%f\n' \
      | sed 's/_block.*//' \
      | sort -u \
      | grep -Ev "$SKIP_REGEX"
  )
fi

if [ "${#SUBJECTS[@]}" -eq 0 ]; then
  echo "ERROR: no subjects to batch" >&2
  exit 1
fi

n_subjects=${#SUBJECTS[@]}
n_batches=$(( (n_subjects + SUBJECTS_PER_JOB - 1) / SUBJECTS_PER_JOB ))

mkdir -p "$BATCH_DIR"
# Drop stale files so a smaller batch count cannot leave old batch_NNN.txt behind.
rm -f "$BATCH_DIR"/batch_*.txt

batch_idx=0
for ((i = 0; i < n_subjects; i += SUBJECTS_PER_JOB)); do
  batch_idx=$((batch_idx + 1))
  batch_file=$(printf '%s/batch_%03d.txt' "$BATCH_DIR" "$batch_idx")
  batch=("${SUBJECTS[@]:i:SUBJECTS_PER_JOB}")
  printf '%s\n' "${batch[@]}" > "$batch_file"
done

# Remember size so submit_all_subjects.sh can detect --batch-size changes.
printf '%s\n' "$SUBJECTS_PER_JOB" > "$BATCH_DIR/.batch_size"

echo "Subjects:  ${n_subjects} → ${SUBJECTS[*]}"
echo "Batch size: ${SUBJECTS_PER_JOB}"
echo "Wrote ${n_batches} batch file(s) under ${BATCH_DIR}/"
ls -1 "$BATCH_DIR"/batch_*.txt
