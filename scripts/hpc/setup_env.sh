#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/hpc/env.sh
echo "runtime: $CONTAINER | image: $SIF | overlay: $OVERLAY"
[[ -f "$SIF" ]] || { echo "No CUDA image found in /share/apps/images; set SIF=..." >&2; exit 1; }
[[ -n "$CONTAINER" ]] || { echo "Neither apptainer nor singularity is installed" >&2; exit 1; }

if [[ ! -f "$OVERLAY" ]]; then
  mkdir -p "$(dirname "$OVERLAY")"
  cp "$OVERLAY_TEMPLATE" "$OVERLAY.gz"
  gunzip "$OVERLAY.gz"
fi

INSTALLER="$PWD/.cache/Miniforge3-Linux-x86_64.sh"
if [[ ! -s "$INSTALLER" ]]; then
  mkdir -p "$(dirname "$INSTALLER")"
  URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  if command -v curl >/dev/null; then curl -fsSL -o "$INSTALLER" "$URL"; else wget -qO "$INSTALLER" "$URL"; fi
fi

"$CONTAINER" exec $RW_FLAGS --overlay "$OVERLAY:rw" "$SIF" /bin/bash -c '
set -euo pipefail
if [[ ! -d /ext3/miniforge3 ]]; then
  bash "$0" -b -p /ext3/miniforge3
fi
cat > /ext3/env.sh <<"ENV"
#!/bin/bash
unset -f which
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
ENV
source /ext3/env.sh
pip install --quiet --upgrade pip
pip install --quiet "torch>=2.2" "torchvision>=0.17" numpy pandas pillow pyyaml tqdm \
  matplotlib jupyter torchsummary pytest
python -c "import torch, torchvision; print(\"torch\", torch.__version__, \"| torchvision\", torchvision.__version__)"
' "$INSTALLER"
"$CONTAINER" exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c \
  'source /ext3/env.sh; PYTHONPATH=src python -m pytest -q -p no:cacheprovider'
echo "Environment ready in $OVERLAY"
