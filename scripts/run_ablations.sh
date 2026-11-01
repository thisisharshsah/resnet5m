#!/bin/bash
# Ablation study for the report: one change at a time, 60 epochs each.
# Results land in runs/abl_*/summary.json and are compared in the notebook.
#
#   bash scripts/run_ablations.sh                                    # locally, one by one
#   LAUNCHER="sbatch --account=<acct> scripts/hpc/train.sbatch" bash scripts/run_ablations.sh
#
# The distillation pair needs a finished teacher (runs/teacher/best.pt).
set -euo pipefail
cd "$(dirname "$0")/.."
LAUNCHER="${LAUNCHER:-python -m cifar_resnet.train --config}"
EPOCHS="${EPOCHS:-60}"

run() {  # run <name> <config> [overrides...]
  local name="$1" config="$2"
  shift 2
  echo ">>> $name"
  $LAUNCHER "$config" run_name="abl_$name" train.epochs="$EPOCHS" "$@"
}

# Architecture, same baseline recipe
run arch_resnet18_narrow configs/resnet18_narrow.yaml
run arch_wide3           configs/wide3.yaml
run arch_deep3           configs/deep3.yaml
run arch_wide3_wider     configs/wide3.yaml "model.channels=[96,192,320]"
run arch_wide3_skip3     configs/wide3.yaml model.skip_kernel_size=3

# Training recipe, wide3 architecture
run aug_basic            configs/wide3.yaml augment.policy=none augment.erasing_prob=0
run aug_trivialaugment   configs/wide3.yaml augment.policy=trivialaugment
run aug_cutmix           configs/wide3.yaml augment.mix=cutmix
run no_label_smoothing   configs/wide3.yaml optim.label_smoothing=0
run no_ema               configs/wide3.yaml train.ema_decay=0
run sam                  configs/wide3.yaml optim.sam_rho=0.05

# Distillation, student architecture and recipe
if [[ -f runs/teacher/best.pt ]]; then
  run student_kd         configs/student.yaml
  run student_no_kd      configs/student.yaml distill.teacher=null
else
  echo "Skipping distillation ablations: train configs/teacher.yaml first."
fi
