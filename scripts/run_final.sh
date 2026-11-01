#!/bin/bash
# Final model on a local GPU: retrain on all 50k training images (no validation hold-out)
# with the chosen config, then write the Kaggle submission. On NYU HPC use
# scripts/hpc/pipeline.sh instead.
#
#   bash scripts/run_final.sh configs/student.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/student.yaml}"
NAME="$(basename "$CONFIG" .yaml)_full"
shift || true
python -m cifar_resnet.train --config "$CONFIG" data.val_size=0 run_name="$NAME" "$@"
python -m cifar_resnet.predict --checkpoint "runs/$NAME/best.pt" --output "submissions/$NAME.csv"
