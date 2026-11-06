#!/bin/bash
#SBATCH --job-name=cifar-pair
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hpc/env.sh
mkdir -p logs

usage="usage: sbatch scripts/hpc/train_pair.sbatch <config> [overrides...] -- <config> [overrides...]"
first=()
second=()
target=first
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then target=second; continue; fi
  if [[ $target == first ]]; then first+=("$arg"); else second+=("$arg"); fi
done
[[ ${#first[@]} -gt 0 && ${#second[@]} -gt 0 ]] || { echo "$usage" >&2; exit 1; }

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 60 \
  > "logs/gpu-${SLURM_JOB_ID:-local}.csv" &
GPU_LOGGER=$!
trap 'kill $GPU_LOGGER 2>/dev/null || true' EXIT

WORKERS=$(( ${SLURM_CPUS_PER_TASK:-24} / 2 - 1 ))

launch() {
  local gpu=$1
  shift
  CUDA_VISIBLE_DEVICES=$gpu "$CONTAINER" exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c '
source /ext3/env.sh
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
python -m cifar_resnet.train --config "$0" train.auto_resume=true "$@"
' "$@" "data.num_workers=$WORKERS"
}

launch 0 "${first[@]}" > "logs/${SLURM_JOB_ID:-local}-gpu0.out" 2>&1 &
a=$!
launch 1 "${second[@]}" > "logs/${SLURM_JOB_ID:-local}-gpu1.out" 2>&1 &
b=$!

status_a=0
status_b=0
wait $a || status_a=$?
wait $b || status_b=$?
echo "gpu0 exit $status_a | gpu1 exit $status_b"
if [[ $status_a -eq 0 && $status_b -eq 0 ]]; then
  scancel --user="$USER" --name="${SLURM_JOB_NAME:-cifar-pair}" --state=PENDING 2>/dev/null || true
fi
exit $(( status_a || status_b ))
