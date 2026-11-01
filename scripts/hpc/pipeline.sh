#!/bin/bash
# Submit the full training pipeline with Slurm dependencies:
#
#   wide3 baseline  ───────────────> baseline submission (calibrates val acc vs leaderboard)
#   teacher ──┬──> student       (5k validation split: the number we trust)
#             └──> student_full  (all 50k images) ──> final submission
#
#   bash scripts/hpc/pipeline.sh <slurm_account>
set -euo pipefail
ACCOUNT="${1:?usage: bash scripts/hpc/pipeline.sh <slurm_account>}"
cd "$(dirname "$0")/../.."
SUBMIT="sbatch --parsable --account=$ACCOUNT"

base=$($SUBMIT scripts/hpc/train.sbatch configs/wide3.yaml)
base_pred=$($SUBMIT --dependency=afterok:$base scripts/hpc/predict.sbatch runs/wide3/best.pt)
teacher=$($SUBMIT --time=12:00:00 scripts/hpc/train.sbatch configs/teacher.yaml)
student=$($SUBMIT --time=10:00:00 --dependency=afterok:$teacher \
  scripts/hpc/train.sbatch configs/student.yaml)
full=$($SUBMIT --time=10:00:00 --dependency=afterok:$teacher \
  scripts/hpc/train.sbatch configs/student.yaml data.val_size=0 run_name=student_full)
full_pred=$($SUBMIT --dependency=afterok:$full scripts/hpc/predict.sbatch runs/student_full/best.pt)

cat <<MSG
Submitted:
  wide3 baseline  $base  -> submission job $base_pred
  teacher         $teacher
  student         $student   (after teacher)
  student_full    $full   (after teacher) -> submission job $full_pred
Watch with: squeue -u \$USER    Logs: logs/
MSG
