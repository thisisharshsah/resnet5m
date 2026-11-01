#!/bin/bash
# Train one config on a single GPU on NYU Torch.
#
#   sbatch --account=<acct> scripts/hpc/train.sbatch configs/wide3.yaml
#   sbatch --account=<acct> --time=10:00:00 scripts/hpc/train.sbatch configs/teacher.yaml
#   sbatch --account=<acct> scripts/hpc/train.sbatch configs/student.yaml data.val_size=0 run_name=student_full
#
# Torch requires --account (find yours with `my_slurm_accounts`) and asks you not to set a
# partition. Jobs with low GPU utilization are cancelled automatically, so this job asks for
# 16 CPU cores to keep the data pipeline ahead of the GPU, and logs utilization to logs/.
# If the job is preempted or requeued, training resumes from the last epoch.
#SBATCH --job-name=cifar-resnet
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hpc/env.sh
mkdir -p logs
module purge 2>/dev/null || true

CONFIG="${1:?usage: sbatch scripts/hpc/train.sbatch <config.yaml> [overrides...]}"
shift

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 \
  > "logs/gpu-${SLURM_JOB_ID:-local}.csv" &
GPU_LOGGER=$!
trap 'kill $GPU_LOGGER 2>/dev/null || true' EXIT

# Arguments are passed positionally so overrides like model.channels=[96,192,320] survive quoting.
apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c '
source /ext3/env.sh
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
python -m cifar_resnet.train --config "$0" train.auto_resume=true "$@"
' "$CONFIG" "data.num_workers=${SLURM_CPUS_PER_TASK:-16}" "$@"
