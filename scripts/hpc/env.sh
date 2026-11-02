export ACCOUNT="${ACCOUNT:-cs_gy_6923-2026fa}"
export PROJECT_DIR="${PROJECT_DIR:-/scratch/$USER/resnet5m}"
export OVERLAY="${OVERLAY:-/scratch/$USER/overlays/cifar-resnet.ext3}"
export OVERLAY_TEMPLATE="${OVERLAY_TEMPLATE:-/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz}"
if [[ -z "${SIF:-}" ]]; then
  SIF="$(ls -1 /share/apps/images/cuda*ubuntu*.sif 2>/dev/null | sort -V | tail -1)"
  export SIF
fi
export CONTAINER="${CONTAINER:-$(command -v apptainer || command -v singularity)}"
if [[ "$(basename "${CONTAINER:-}")" == apptainer ]]; then RW_FLAGS="--fakeroot"; else RW_FLAGS=""; fi
export RW_FLAGS
