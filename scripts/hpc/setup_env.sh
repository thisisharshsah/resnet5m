#!/bin/bash
# One-time Python environment for this project on NYU Torch.
# Run it inside an interactive job (installs are too heavy for login nodes):
#
#   srun --cpus-per-task=4 --mem=16GB --time=01:00:00 --account=<your_account> --pty /bin/bash
#   bash scripts/hpc/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ ! -f "$OVERLAY" ]]; then
  mkdir -p "$(dirname "$OVERLAY")"
  cp "$OVERLAY_TEMPLATE" "$OVERLAY.gz"
  gunzip "$OVERLAY.gz"
fi

apptainer exec --fakeroot --overlay "$OVERLAY:rw" "$SIF" /bin/bash -c '
set -euo pipefail
if [[ ! -d /ext3/miniforge3 ]]; then
  cd /tmp
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
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
'
echo "Environment ready in $OVERLAY"
