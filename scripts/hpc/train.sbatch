#!/bin/bash
#SBATCH --job-name=cifar-resnet
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hpc/env.sh
mkdir -p logs

CONFIG="${1:?usage: sbatch scripts/hpc/train.sbatch <config.yaml> [overrides...]}"
shift

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 60 \
  > "logs/gpu-${SLURM_JOB_ID:-local}.csv" &
GPU_LOGGER=$!
trap 'kill $GPU_LOGGER 2>/dev/null || true' EXIT

"$CONTAINER" exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c '
source /ext3/env.sh
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
python -m cifar_resnet.train --config "$0" train.auto_resume=true "$@"
' "$CONFIG" "data.num_workers=$(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))" "$@"
