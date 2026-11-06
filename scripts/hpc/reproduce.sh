#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

STAGE="${1:-all}"
RETRIES="${RETRIES:-8}"
SUBMIT="sbatch --parsable"
TRAIN=scripts/hpc/train.sbatch
PREDICT=scripts/hpc/predict.sbatch
TEACHERS=(teacher teacher_s1 teacher_s2 teacher_s3 teacher_s4)
TEACHER_IDS=""

after_teachers() {
  if [[ -n "$TEACHER_IDS" ]]; then
    echo "--dependency=afterok:$TEACHER_IDS"
    return
  fi
  for t in "${TEACHERS[@]}"; do
    [[ -f "runs/$t/best.pt" ]] || { echo "runs/$t/best.pt is missing, run the teachers stage first" >&2; exit 1; }
  done
}

with_full() {
  local config=$1 name=$2 dep val full pred
  dep=$(after_teachers)
  val=$($SUBMIT --time=12:00:00 $dep "$TRAIN" "$config")
  full=$($SUBMIT --time=12:00:00 $dep "$TRAIN" "$config" data.val_size=0 run_name="${name}_full")
  pred=$($SUBMIT --dependency="afterok:$full" "$PREDICT" "runs/${name}_full/best.pt")
  echo "$name $val | ${name}_full $full | submission $pred"
}

baselines() {
  local wide3 pred nokd
  wide3=$($SUBMIT "$TRAIN" configs/wide3.yaml)
  pred=$($SUBMIT --dependency="afterok:$wide3" "$PREDICT" runs/wide3/best.pt)
  nokd=$($SUBMIT "$TRAIN" configs/student.yaml distill.teacher=null run_name=student_no_kd)
  echo "wide3 $wide3 | submission $pred | student_no_kd $nokd"
}

teachers() {
  local ids=()
  ids+=("$($SUBMIT --time=12:00:00 "$TRAIN" configs/teacher.yaml)")
  ids+=("$($SUBMIT --time=12:00:00 "$TRAIN" configs/teacher.yaml train.seed=1 run_name=teacher_s1)")
  ids+=("$($SUBMIT --time=12:00:00 "$TRAIN" configs/teacher.yaml train.seed=2 run_name=teacher_s2)")
  ids+=("$($SUBMIT --time=12:00:00 "$TRAIN" configs/teacher.yaml train.epochs=400 train.seed=3 run_name=teacher_s3)")
  ids+=("$($SUBMIT --time=12:00:00 "$TRAIN" configs/teacher.yaml train.epochs=400 train.seed=4 augment.policy=trivialaugment run_name=teacher_s4)")
  TEACHER_IDS=$(IFS=:; echo "${ids[*]}")
  echo "teachers ${ids[*]}"
}

students() {
  with_full configs/student.yaml student
  with_full configs/student_long.yaml student_long
}

final() {
  local dep=singleton ids=() pred
  after_teachers > /dev/null
  [[ -n "$TEACHER_IDS" ]] && dep="afterok:$TEACHER_IDS,singleton"
  for _ in $(seq 1 "$RETRIES"); do
    ids+=("$($SUBMIT --dependency="$dep" scripts/hpc/train_pair.sbatch \
      configs/student_v5.yaml data.val_size=0 run_name=student_v5_full -- configs/student_v5.yaml)")
  done
  pred=$($SUBMIT --dependency="afterany:$(IFS=:; echo "${ids[*]}")" "$PREDICT" runs/student_v5_full/best.pt)
  echo "student_v5 and student_v5_full ${ids[*]} | submission $pred"
}

case "$STAGE" in
  baselines) baselines ;;
  teachers) teachers ;;
  students) students ;;
  final) final ;;
  all) baselines; teachers; students; final ;;
  *) echo "usage: bash scripts/hpc/reproduce.sh [baselines|teachers|students|final|all]" >&2; exit 1 ;;
esac
