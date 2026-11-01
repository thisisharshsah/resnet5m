#!/bin/bash
# Write a Kaggle submission from a trained checkpoint.
#
#   sbatch --account=<acct> scripts/hpc/predict.sbatch runs/student_full/best.pt
#SBATCH --job-name=cifar-predict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hpc/env.sh
mkdir -p logs submissions
CKPT="${1:?usage: sbatch scripts/hpc/predict.sbatch <checkpoint>}"
NAME="$(basename "$(dirname "$CKPT")")"

apptainer exec --nv --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c '
source /ext3/env.sh
export PYTHONPATH="$PWD/src"
python -m cifar_resnet.predict --checkpoint "$0" --output "submissions/$1.csv"
' "$CKPT" "$NAME"
