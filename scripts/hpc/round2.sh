#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SUBMIT="sbatch --parsable"

t1=$($SUBMIT --time=12:00:00 scripts/hpc/train.sbatch configs/teacher.yaml train.seed=1 run_name=teacher_s1)
t2=$($SUBMIT --time=12:00:00 scripts/hpc/train.sbatch configs/teacher.yaml train.seed=2 run_name=teacher_s2)
echo "teacher_s1 $t1 | teacher_s2 $t2"
[[ "${1:-}" == "teachers" ]] && exit 0

[[ -f runs/teacher/best.pt ]] || { echo "runs/teacher/best.pt missing: run the pipeline first" >&2; exit 1; }
dep="--dependency=afterok:$t1:$t2"
long=$($SUBMIT --time=12:00:00 $dep scripts/hpc/train.sbatch configs/student_long.yaml)
full=$($SUBMIT --time=12:00:00 $dep scripts/hpc/train.sbatch configs/student_long.yaml data.val_size=0 run_name=student_long_full)
pred=$($SUBMIT --dependency=afterok:$full scripts/hpc/predict.sbatch runs/student_long_full/best.pt)
echo "student_long $long | student_long_full $full -> submission job $pred"
