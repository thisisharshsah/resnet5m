# Shared settings for the NYU Torch scripts. Override any of these in your shell.
# Paths follow NYU's "Apptainer with Conda" guide: Python lives in an ext3 overlay so the
# many small conda files don't count against your file quota.
export PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/cifar10-resnet}"
export OVERLAY="${OVERLAY:-/scratch/$USER/overlays/cifar-resnet.ext3}"
export OVERLAY_TEMPLATE="${OVERLAY_TEMPLATE:-/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz}"
export SIF="${SIF:-/share/apps/images/cuda-13.3.1-ubuntu-26.04.sif}"
