# Parameter-efficient ResNets for CIFAR-10 (under 5M parameters)

NYU Tandon Deep Learning, Fall 2026, Project 1. The goal is the highest accuracy on a held-out
CIFAR-10 evaluation set ([Kaggle competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl))
using a ResNet trained from scratch with **at most 5,000,000 trainable parameters**.

## Results

| Run | Params | Val acc (5k held-out) | Kaggle public | Kaggle private |
|---|---|---|---|---|
| _pending training runs_ | | | | |

Final accuracy and parameter count are printed at the end of
[`notebooks/project1.ipynb`](notebooks/project1.ipynb).

## Approach

The evaluation images are not the standard CIFAR-10 test set: their color statistics differ
and leaderboard scores sit several points below typical CIFAR-10 accuracy, which suggests a
newly collected CIFAR-style test set. On such sets, accuracy tracks the original test accuracy
almost linearly, so the strategy is to maximize genuine CIFAR-10 accuracy rather than tune to
the leaderboard.

**Architecture.** A configurable ResNet whose residual blocks compute `ReLU(S(x) + F(x))`, with
`F` = conv, BN, ReLU, conv, BN and `S` the identity or a KxK conv + BN projection. Stage widths
`C_i`, filter size `F_i`, shortcut kernel `K_i`, pool size `P`, blocks per stage,
squeeze-and-excitation and dropout are all set from YAML.

| Config | Role | Stages (channels x blocks) | Params |
|---|---|---|---|
| `configs/student.yaml` | main submission | 96x2, 192x2, 320x2 | 4,902,506 |
| `configs/wide3.yaml` | baseline | 80x2, 160x2, 320x2 | 4,337,370 |
| `configs/resnet18_narrow.yaml` | ablation | 42x2, 84x2, 168x2, 336x2 (ResNet-18 layout) | 4,815,940 |
| `configs/deep3.yaml` | ablation | 64x3, 128x3, 256x3 | 4,327,754 |
| `configs/teacher.yaml` | teacher only, never submitted | 160x4, 320x4, 640x4 (WRN-28-10 scale) | 36,690,090 |

**Training recipe.** SGD with Nesterov momentum (lr 0.1, batch 128, weight decay 5e-4 on conv
and linear weights only), 5-epoch warmup then cosine decay, random crop with reflect padding,
horizontal flip, AutoAugment (CIFAR-10 policy), random erasing, an exponential moving average
of the weights, and flip test-time augmentation. The student adds:

- **Sharpness-Aware Minimization** (rho 0.05), which seeks flat minima and tends to generalize
  better under distribution shift. BatchNorm running statistics are frozen during SAM's second
  pass so they are updated once per step.
- **Knowledge distillation** from a WRN-28-10-scale teacher trained from scratch on the same
  45k images (alpha 0.9, temperature 4). The teacher sees exactly the student's augmented
  input ("consistent teaching") and is trained without label smoothing, which transfers more
  knowledge.
- **CutMix** on half the batches and a 300-epoch schedule.

**Model selection.** A stratified 5,000-image validation split is held out from the 50,000
training images and used for every decision. The chosen recipe is then retrained on all
50,000 images for the final submission. The public leaderboard (about 1,000 images) is only a
sanity check: differences under about 1% there are noise.

## Rules compliance

Every rule in the project brief, and how this repository guarantees it:

| Rule | How it is enforced |
|---|---|
| At most 5M trainable parameters | `build_model` refuses over-budget configs; `cifar-predict` refuses any model over the hard 5,000,000 limit (`rules.COMPETITION_MAX_PARAMS`, not configurable); a test checks every submission config |
| One model per submission | `cifar-predict` takes exactly one checkpoint; there is no ensembling code |
| Train from scratch, no pretrained weights | Models are built from random init; every checkpoint is stamped with its origin and only our own checkpoints are accepted as teachers or for prediction (`rules.check_trained_from_scratch`) |
| No external datasets | Data loading reads only the competition's `train_labels.csv` and `train_images/` |
| Evaluation images never used for training, fine-tuning or model selection | Training code only ever loads the `train` split (a test asserts the test split is never read during training); prediction runs in eval mode, so test images never update weights or BatchNorm statistics; no pseudo-labeling or test-time adaptation; model selection uses our own validation split |
| Distillation teachers trained from scratch on the permitted data | `rules.check_teacher` rejects foreign checkpoints and teachers that saw the student's validation images |
| Reproducible preprocessing and training | Fixed seeds, deterministic validation split, resolved config saved with every run |
| Citations and AI-assistance disclosure | See References and Acknowledgements |

## Repository layout

```
configs/                 experiment configs (YAML); every field documented in src/cifar_resnet/config.py
src/cifar_resnet/
    models/resnet.py     residual block and configurable ResNet
    data.py              data loading, uint8 cache, augmentation, stratified split
    engine.py            train/eval loops, optimizer, LR schedule, MixUp/CutMix, distillation
    sam.py               Sharpness-Aware Minimization
    rules.py             competition rules enforced in code
    train.py             training entry point (cifar-train)
    predict.py           Kaggle submission writer (cifar-predict)
    params.py            parameter counts per config (cifar-params)
    ema.py, checkpoint.py, utils.py
notebooks/project1.ipynb results notebook (generated by notebooks/build_notebook.py)
scripts/hpc/             NYU Torch: environment setup, train/predict jobs, full pipeline
scripts/                 ablation sweep, local final retrain
tests/                   unit, rules and end-to-end tests on synthetic data
```

Each run writes to `runs/<run_name>/`: the resolved `config.yaml`, `train.log`, per-epoch
`metrics.csv`, `summary.json`, and checkpoints (`best.pt`, `last.pt`, `final.pt`). Checkpoints
store their config, so any model can be rebuilt from the file alone.

## Setup

Requires Python 3.10+ and PyTorch 2.2+. Training uses CUDA if available, then Apple Silicon
(MPS), then CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook,dev]"
pytest                       # about a minute on CPU, no data needed
```

### Data

Download the competition data from the Kaggle **Data** tab into `data/`:

```
data/
    train_labels.csv         id, filename, label
    test.csv                 id, filename
    sample_submission.csv
    train_images/            50,000 PNGs
    test_images/             2,000 PNGs
```

`data/` is git-ignored. The first run decodes the PNGs into a uint8 cache under `.cache/data/`
(about 15 seconds); later runs start immediately.

## Training on NYU HPC (Torch)

Torch needs the NYU VPN (or campus network) and Duo. Find your Slurm account with
`my_slurm_accounts`.

```bash
# 1. From your Mac: copy the project (code + data) to scratch
cd "<folder containing cifar10-resnet>"
tar czf - --exclude=.cache cifar10-resnet | ssh <netid>@dtn.torch.hpc.nyu.edu "tar xzf - -C /scratch/<netid>"

# 2. On Torch, once: build the Python environment in an Apptainer overlay
ssh <netid>@login.torch.hpc.nyu.edu
cd /scratch/$USER/cifar10-resnet
srun --account=<account> --cpus-per-task=4 --mem=16GB --time=01:00:00 --pty /bin/bash
bash scripts/hpc/setup_env.sh && exit

# 3. Submit everything: baseline, teacher, then student and the final full-data student
bash scripts/hpc/pipeline.sh <account>
squeue -u $USER
```

Single runs: `sbatch --account=<account> scripts/hpc/train.sbatch <config> [overrides]`.
Torch cancels jobs with low GPU utilization, so jobs request 16 CPU cores for the data
pipeline and log GPU utilization to `logs/gpu-<job>.csv`. Jobs resume from their last epoch
if preempted or requeued.

## Usage (any machine)

```bash
cifar-params configs/*.yaml --summary                      # parameter breakdown per config
cifar-train --config configs/wide3.yaml                    # baseline with a 5k validation split
cifar-train --config configs/teacher.yaml                  # teacher (needs a GPU)
cifar-train --config configs/student.yaml                  # distilled student
cifar-train --config configs/student.yaml train.epochs=50 run_name=quick   # any field can be overridden
cifar-train --resume runs/student/last.pt                  # continue an interrupted run
cifar-predict --checkpoint runs/student_full/best.pt --output submissions/student_full.csv
```

**Ablations.** `bash scripts/run_ablations.sh` (or with
`LAUNCHER="sbatch --account=<account> scripts/hpc/train.sbatch"` on Torch) runs one change at a
time at 60 epochs: architecture, augmentation policy, CutMix, label smoothing, EMA, SAM, and
distillation on/off. The notebook compares all runs from their `summary.json` files.

## Reproducibility

Seeds are fixed (`train.seed`), the validation split is deterministic, and every run saves its
resolved config next to its results. Small run-to-run differences remain on GPU because cuDNN
autotuning is enabled for speed.

## References

- K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
- S. Zagoruyko, N. Komodakis. Wide Residual Networks. BMVC 2016.
- P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. Sharpness-Aware Minimization for Efficiently
  Improving Generalization. ICLR 2021. Implementation adapted from
  https://github.com/davda54/sam (MIT).
- J. Kwon et al. ASAM: Adaptive Sharpness-Aware Minimization. ICML 2021.
- G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. 2015.
- L. Beyer et al. Knowledge Distillation: A Good Teacher is Patient and Consistent. CVPR 2022.
- R. Muller, S. Kornblith, G. Hinton. When Does Label Smoothing Help? NeurIPS 2019.
- E. D. Cubuk et al. AutoAugment: Learning Augmentation Policies from Data. CVPR 2019.
- T. DeVries, G. Taylor. Improved Regularization of CNNs with Cutout. 2017.
- Z. Zhong et al. Random Erasing Data Augmentation. AAAI 2020.
- S. Yun et al. CutMix. ICCV 2019. H. Zhang et al. mixup. ICLR 2018.
- C. Szegedy et al. Rethinking the Inception Architecture (label smoothing). CVPR 2016.
- P. Goyal et al. Accurate, Large Minibatch SGD (zero-init residual BN, warmup). 2017.
- J. Hu et al. Squeeze-and-Excitation Networks. CVPR 2018.
- B. T. Polyak, A. B. Juditsky. Acceleration of Stochastic Approximation by Averaging. 1992.
- B. Recht et al. Do CIFAR-10 Classifiers Generalize to CIFAR-10? 2018.
- K. Liu. pytorch-cifar, https://github.com/kuangliu/pytorch-cifar (reference training setup).

## Acknowledgements

Parts of this codebase were written with the help of an AI assistant (Claude, Anthropic), as
permitted by the course policy. All experiments, results and design decisions were run and
verified by the team.
