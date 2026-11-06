#!/bin/bash
#SBATCH --job-name=cifar-eval
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hpc/env.sh
mkdir -p logs
[[ $# -gt 0 ]] || { echo "usage: sbatch scripts/hpc/eval.sbatch <checkpoint>..." >&2; exit 1; }

"$CONTAINER" exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c '
source /ext3/env.sh
export PYTHONPATH="$PWD/src"
python -m cifar_resnet.evaluate --checkpoint "$@"
' _ "$@"
