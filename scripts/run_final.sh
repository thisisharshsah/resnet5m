#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/student.yaml}"
NAME="$(basename "$CONFIG" .yaml)_full"
shift || true
python -m cifar_resnet.train --config "$CONFIG" data.val_size=0 run_name="$NAME" "$@"
python -m cifar_resnet.predict --checkpoint "runs/$NAME/best.pt" --output "submissions/$NAME.csv"
