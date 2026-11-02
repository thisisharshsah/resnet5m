#!/bin/bash
#SBATCH --job-name=cifar-predict
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hpc/env.sh
mkdir -p logs submissions
CKPT="${1:?usage: sbatch scripts/hpc/predict.sbatch <checkpoint>}"
NAME="$(basename "$(dirname "$CKPT")")"

"$CONTAINER" exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c '
source /ext3/env.sh
export PYTHONPATH="$PWD/src"
python -m cifar_resnet.predict --checkpoint "$0" --output "submissions/$1.csv"
' "$CKPT" "$NAME"
