# resnet5m

A ResNet under 5M parameters for CIFAR-10, trained from scratch.

NYU Tandon Deep Learning, Fall 2026, Project 1, Kaggle competition
[NYU Tandon Deep Learning F26](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl).
The final model has 4,902,506 parameters, reaches 97.46% on a 5,000 image validation split
and 0.899 on the public leaderboard.

## Results

| Run | Params | Val acc (flip TTA) | Kaggle public | Kaggle private |
|---|---|---|---|---|
| `student_v5` / `student_v5_full` (final) | 4,902,506 | 97.46% (97.48%) | 0.899 | after Nov 6 |
| `student_long` / `student_long_full` | 4,902,506 | 97.18% (97.20%) | 0.896 | |
| `student` / `student_full` | 4,902,506 | 97.14% (97.12%) | 0.887 | |
| `student_no_kd` (`student` without a teacher) | 4,902,506 | 96.86% (96.88%) | | |
| `wide3` (baseline) | 4,337,370 | 96.10% (96.44%) | 0.875 | |

Val accuracy comes from the run trained on 45k images, the Kaggle score from its twin
retrained on all 50k. The teachers (36.7M params) reach 97.16% to 97.74% and are never
submitted. `notebooks/project1.ipynb` rebuilds every table and plot from `runs/` and prints the
final accuracy and parameter count, no GPU needed.

## Quick start

Python 3.10+ and PyTorch 2.2+. Uses CUDA if there is one, then Apple MPS, then CPU.

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test
make params
make notebook
```

Put the Kaggle data in `data/`:

```
data/train_labels.csv
data/test.csv
data/sample_submission.csv
data/train_images/
data/test_images/
```

The first run decodes the PNGs into `.cache/` (about 15 s), later runs start right away.

## How it works

The model is a plain ResNet with `ReLU(F(x) + S(x))` blocks, where `F` is conv, BN, ReLU,
conv, BN and `S` is the identity or a 1x1 conv projection. Widths, depth, kernel sizes and
pooling are set in YAML.

| Config | Role | Stages | Params |
|---|---|---|---|
| `student_v5.yaml` | final model, 5 teachers, 1600 epochs | 96x2, 192x2, 320x2 | 4,902,506 |
| `teacher.yaml` | teacher, never submitted | 160x4, 320x4, 640x4 | 36,690,090 |
| `student_long.yaml` | 3 teachers, 800 epochs | 96x2, 192x2, 320x2 | 4,902,506 |
| `student.yaml` | 1 teacher, 300 epochs | 96x2, 192x2, 320x2 | 4,902,506 |
| `wide3.yaml` | baseline, no teacher | 80x2, 160x2, 320x2 | 4,337,370 |
| `resnet18_narrow.yaml` | ablation | 42x2, 84x2, 168x2, 336x2 | 4,815,940 |
| `deep3.yaml` | ablation | 64x3, 128x3, 256x3 | 4,327,754 |
| `smoke.yaml` | tiny config for tests | | |

Every run uses SGD with Nesterov, lr 0.1, batch 128, cosine schedule with 5 warmup epochs,
weight decay 5e-4, random crop, flip, AutoAugment, random erasing, EMA weights and flip TTA at
test time. The final model adds:

- SAM (rho 0.05), flatter minima tend to survive the test set shift better
- knowledge distillation from five teachers trained from scratch on the same split (three for
  200 epochs, two for 400), logits averaged, alpha 0.9, T 4. The student is still one model
  under 5M.
- CutMix on half the batches, 1600 epochs

Model selection uses a 5,000 image validation split taken from the training set. The chosen
config is then retrained on all 50,000 images for the submission. The public leaderboard is
only 1,000 images, so differences under 1% there are noise.

## Reproduce

Locally, one command per run. Any config field can be overridden on the command line, for
example `train.epochs=50` or `model.channels=[96,192,320]`.

```bash
cifar-train --config configs/teacher.yaml
cifar-train --config configs/teacher.yaml train.seed=1 run_name=teacher_s1
cifar-train --config configs/teacher.yaml train.seed=2 run_name=teacher_s2
cifar-train --config configs/teacher.yaml train.epochs=400 train.seed=3 run_name=teacher_s3
cifar-train --config configs/teacher.yaml train.epochs=400 train.seed=4 augment.policy=trivialaugment run_name=teacher_s4
cifar-train --config configs/student_v5.yaml
cifar-eval --checkpoint runs/student_v5/best.pt
bash scripts/run_final.sh configs/student_v5.yaml
```

`run_final.sh` retrains a config on all 50k images and writes the Kaggle csv to `submissions/`.
`bash scripts/run_ablations.sh` runs the 60 epoch ablations, one change at a time.
`cifar-train --resume runs/<run_name>/last.pt` continues an interrupted run.

On NYU HPC Cloud Bursting (account `cs_gy_6923-2026fa`, VPN needed off campus), upload the
project to `/scratch/$USER/resnet5m`, then:

```bash
sbatch scripts/hpc/setup_env.sh
bash scripts/hpc/reproduce.sh all
squeue -u $USER
```

`reproduce.sh` takes one stage at a time too: `baselines`, `teachers`, `students` or `final`.
Training runs on one A100 (`c12m85-a100-1`), setup, evaluation and prediction on the CPU
partition. The final stage trains `student_v5` and `student_v5_full` side by side on a two GPU
node (`train_pair.sbatch`) and queues several identical copies, so a node that fails to boot is
retried. Cloud nodes are stopped after 20 minutes of low GPU use, so jobs give all CPUs to the
data pipeline, log GPU use to `logs/` and resume from their last epoch when requeued.

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
configs/             one YAML per run family, fields in src/cifar_resnet/config.py
src/cifar_resnet/
  models/resnet.py   the ResNet
  models/ensemble.py teacher ensemble for distillation
  data.py            loading, cache, augmentation, split
  engine.py          train and eval loops, optimizer, schedule, mix, distillation, TTA
  sam.py             SAM
  rules.py           competition rules in code
  train.py           cifar-train
  evaluate.py        cifar-eval, val accuracy per TTA mode and confusion matrix
  predict.py         cifar-predict, writes the Kaggle csv
  params.py          cifar-params
runs/<run_name>/     config.yaml, metrics.csv, summary.json, val_eval.json (checkpoints stay local)
submissions/         the csv files sent to Kaggle
notebooks/           project1.ipynb, built by build_notebook.py
scripts/             local ablations and final retrain
scripts/hpc/         NYU Cloud Bursting jobs, reproduce.sh is the entry point
tests/
```

A checkpoint carries its config, so a model can be rebuilt from the `.pt` file alone.

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
