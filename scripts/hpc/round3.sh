#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SUBMIT="sbatch --parsable"
RETRIES="${RETRIES:-8}"
dep="singleton"

if [[ "${1:-}" != "students" ]]; then
  t3=$($SUBMIT --time=12:00:00 scripts/hpc/train.sbatch configs/teacher.yaml train.epochs=400 train.seed=3 run_name=teacher_s3)
  t4=$($SUBMIT --time=12:00:00 scripts/hpc/train.sbatch configs/teacher.yaml train.epochs=400 train.seed=4 augment.policy=trivialaugment run_name=teacher_s4)
  dep="afterok:$t3:$t4,singleton"
  echo "teacher_s3 $t3 | teacher_s4 $t4"
fi

pair_args=(configs/student_v5.yaml data.val_size=0 run_name=student_v5_full -- configs/student_v5.yaml)
ids=()
for _ in $(seq 1 "$RETRIES"); do
  ids+=("$($SUBMIT --dependency="$dep" scripts/hpc/train_pair.sbatch "${pair_args[@]}")")
done
pred=$($SUBMIT --dependency="afterany:$(IFS=:; echo "${ids[*]}")" scripts/hpc/predict.sbatch runs/student_v5_full/best.pt)
echo "student_v5_full + student_v5: ${ids[*]} -> submission job $pred"
