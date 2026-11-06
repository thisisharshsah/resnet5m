# resnet5m

A ResNet under 5M parameters for CIFAR-10.

NYU Tandon Deep Learning, Fall 2026, Project 1. Kaggle competition: train a ResNet from
scratch and get the best accuracy on a hidden CIFAR-10 test set with at most 5M trainable
parameters.

Competition: https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl

## Results

| Run | Params | Val acc (5k held out) | Kaggle public | Kaggle private |
|---|---|---|---|---|
| `student_v5` / `student_v5_full` (final) | 4,902,506 | 97.46% (97.48% flip TTA) | 0.899 | after Nov 6 |
| `student_long` / `student_long_full` | 4,902,506 | 97.18% (97.20%) | 0.896 | |
| `student` / `student_full` | 4,902,506 | 97.14% (97.12%) | 0.887 | |
| `student_no_kd` (same as `student`, no teacher) | 4,902,506 | 96.86% (96.88%) | | |
| `wide3` (baseline) | 4,337,370 | 96.10% (96.44%) | 0.875 | |

Val accuracy comes from the run trained on 45k images, the Kaggle score from its twin retrained on
all 50k images. Teachers (36.7M params) reach 97.16% to 97.74% and are never submitted.

The notebook `notebooks/project1.ipynb` prints the final accuracy and parameter count.

## What we do

The hidden test set is not the normal CIFAR-10 test set (different colour stats, scores a
few points lower than usual), so the plan is just to get the best real CIFAR-10 accuracy we
can and not tune to the leaderboard.

Model: a plain ResNet with `ReLU(S(x) + F(x))` blocks, `F` = conv, BN, ReLU, conv, BN and
`S` = identity or a 1x1 conv projection. Widths, depth, kernel sizes and pooling are all set
in YAML.

| Config | Role | Stages | Params |
|---|---|---|---|
| `student_v5.yaml` | final model, 5 teachers, 1600 epochs | 96x2, 192x2, 320x2 | 4,902,506 |
| `student_long.yaml` | 3 teachers, 800 epochs | 96x2, 192x2, 320x2 | 4,902,506 |
| `student.yaml` | same model, shorter schedule | 96x2, 192x2, 320x2 | 4,902,506 |
| `wide3.yaml` | baseline | 80x2, 160x2, 320x2 | 4,337,370 |
| `resnet18_narrow.yaml` | ablation | 42x2, 84x2, 168x2, 336x2 | 4,815,940 |
| `deep3.yaml` | ablation | 64x3, 128x3, 256x3 | 4,327,754 |
| `teacher.yaml` | teacher only, never submitted | 160x4, 320x4, 640x4 | 36,690,090 |

Training: SGD with Nesterov, lr 0.1, batch 128, cosine schedule with 5 warmup epochs, weight
decay 5e-4, random crop, flip, AutoAugment, random erasing, EMA weights, flip TTA at test
time. The final model adds:

- SAM (rho 0.05), flatter minima tend to survive the test set shift better
- knowledge distillation from five big teachers trained from scratch on the same split
  (three for 200 epochs, two for 400), logits averaged, alpha 0.9, T 4. The student is still
  one model under 5M.
- CutMix on half the batches, 1600 epochs

Model selection uses a 5,000 image validation split taken from the training set. The chosen
config is then retrained on all 50,000 images for the submission. The public leaderboard is
only 1,000 images, so differences under 1% there are noise.

## Rules

| Rule | How it is kept |
|---|---|
| max 5M params | `build_model` rejects bigger configs, `cifar-predict` refuses anything over 5,000,000, a test checks every config |
| one model per submission | `cifar-predict` takes one checkpoint, there is no ensembling at inference |
| from scratch, no pretrained weights | random init only, checkpoints are stamped and only our own are accepted |
| no outside data | the loader reads only `train_labels.csv` and `train_images/` |
| test images never used for training or selection | training only loads the train split (tested), prediction runs in eval mode, no pseudo labels |
| teachers trained from scratch on allowed data | `rules.check_teacher` rejects anything else and teachers that saw the validation split |
| course compute on Cloud Bursting with the course account | all job scripts are pinned to `cs_gy_6923-2026fa` |

## Layout

```
configs/            YAML configs, fields are in src/cifar_resnet/config.py
src/cifar_resnet/
  models/resnet.py  the ResNet
  data.py           loading, cache, augmentation, split
  engine.py         train and eval loops, optimizer, schedule, mix, distillation
  sam.py            SAM
  rules.py          competition rules in code
  train.py          cifar-train
  predict.py        cifar-predict, writes the Kaggle csv
  params.py         cifar-params
notebooks/          project1.ipynb, built by build_notebook.py
scripts/hpc/        NYU Cloud Bursting jobs
scripts/            ablations, local final run
tests/
```

Each run writes to `runs/<run_name>/`: `config.yaml`, `train.log`, `metrics.csv`,
`summary.json`, `best.pt`, `last.pt`, `final.pt`. A checkpoint carries its config, so a model
can be rebuilt from the file alone.

## Setup

Python 3.10+, PyTorch 2.2+. CUDA if there is one, then Apple MPS, then CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook,dev]"
pytest
```

Data goes in `data/` straight from the Kaggle Data tab:

```
data/train_labels.csv
data/test.csv
data/sample_submission.csv
data/train_images/
data/test_images/
```

First run decodes the PNGs into `.cache/` (about 15 s), later runs start right away.

## Run

```bash
cifar-params configs/*.yaml --summary
cifar-train --config configs/wide3.yaml
cifar-train --config configs/teacher.yaml
cifar-train --config configs/teacher.yaml train.seed=1 run_name=teacher_s1
cifar-train --config configs/teacher.yaml train.seed=2 run_name=teacher_s2
cifar-train --config configs/teacher.yaml train.epochs=400 train.seed=3 run_name=teacher_s3
cifar-train --config configs/teacher.yaml train.epochs=400 train.seed=4 augment.policy=trivialaugment run_name=teacher_s4
cifar-train --config configs/student_v5.yaml
cifar-train --config configs/student_v5.yaml data.val_size=0 run_name=student_v5_full
cifar-eval --checkpoint runs/student_v5/best.pt
cifar-predict --checkpoint runs/student_v5_full/best.pt --output submissions/final.csv
cifar-train --resume runs/student_v5/last.pt
```

Any config field can be overridden on the command line, for example `train.epochs=50` or
`model.channels=[96,192,320]`.

Ablations: `bash scripts/run_ablations.sh`, one change at a time for 60 epochs. The notebook
collects all `summary.json` files into one table.

## NYU HPC Cloud Bursting

Account `cs_gy_6923-2026fa`, VPN needed off campus. Upload the project as one archive with
Files > Upload on the portal, then:

```bash
cd /scratch/$USER
sbatch --account=cs_gy_6923-2026fa --partition=n2c48m24 -c 4 --mem=8G -t 01:00:00 --wrap "tar xzf resnet5m.tar.gz"
cd resnet5m
sbatch scripts/hpc/setup_env.sh
bash scripts/hpc/pipeline.sh
bash scripts/hpc/round2.sh
bash scripts/hpc/round3.sh
squeue -u $USER
```

Training runs on one A100 (`c12m85-a100-1`), setup and prediction on the CPU partition.
`round3.sh` trains the final student and its validation twin side by side on a two GPU node
(`c24m170-a100-2`, `train_pair.sbatch`) and queues several identical copies, so a node that fails
to boot is retried automatically.
Cloud instances get killed after 20 minutes of low GPU use, so jobs give all CPUs to the data
pipeline and log GPU use to `logs/`. Jobs resume from their last epoch if requeued.

## Config fields

| Field | Default | Meaning |
|---|---|---|
| `data.val_size` | 5000 | held out images, 0 trains on everything |
| `data.split_seed` | 42 | which images are held out |
| `augment.policy` | autoaugment | none, autoaugment, trivialaugment, randaugment |
| `augment.mix` | none | none, mixup, cutmix |
| `model.channels`, `model.blocks` | | width and depth per stage |
| `model.kernel_size`, `model.skip_kernel_size` | 3, 1 | conv and shortcut kernel sizes |
| `model.pool_size` | null | avg pool size, null means global |
| `optim.sam_rho` | 0 | SAM radius, 0 is off |
| `train.ema_decay` | 0.999 | weight EMA, 0 is off |
| `train.auto_resume` | false | continue from `last.pt` if it exists |
| `distill.teacher` | null | one checkpoint or a list |

## References

- He et al., Deep Residual Learning for Image Recognition, CVPR 2016
- Zagoruyko and Komodakis, Wide Residual Networks, BMVC 2016
- Foret et al., Sharpness-Aware Minimization, ICLR 2021. SAM code adapted from https://github.com/davda54/sam (MIT)
- Hinton, Vinyals and Dean, Distilling the Knowledge in a Neural Network, 2015
- Beyer et al., Knowledge Distillation: A Good Teacher is Patient and Consistent, CVPR 2022
- Muller, Kornblith and Hinton, When Does Label Smoothing Help?, NeurIPS 2019
- Cubuk et al., AutoAugment, CVPR 2019
- DeVries and Taylor, Cutout, 2017. Zhong et al., Random Erasing, AAAI 2020
- Yun et al., CutMix, ICCV 2019. Zhang et al., mixup, ICLR 2018
- Goyal et al., Accurate, Large Minibatch SGD, 2017
- Recht et al., Do CIFAR-10 Classifiers Generalize to CIFAR-10?, 2018
- kuangliu/pytorch-cifar, https://github.com/kuangliu/pytorch-cifar

Parts of this code were written with help from an AI assistant (Claude), which the course
allows as long as it is acknowledged. All experiments were run and checked by us.
