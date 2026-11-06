from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        "# resnet5m: a ResNet under 5M parameters for CIFAR-10\n\n"
        "Everything here is read from the run folders produced by the CLI (`runs/<name>/`: "
        "`summary.json`, `metrics.csv`, `val_eval.json`), so the notebook needs no GPU and no "
        "checkpoints. Training commands are in the README."
    ),
    md("## 1. Setup"),
    code(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "os.chdir(ROOT)\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n\n"
        "from cifar_resnet.config import load_config\n"
        "from cifar_resnet.data import CLASSES, load_split\n"
        "from cifar_resnet.models import build_model, count_parameters\n"
        "from cifar_resnet.params import layer_table\n\n"
        "FINAL_CONFIG = 'configs/student_v5.yaml'\n"
        "FINAL_RUN = 'student_v5'\n"
        "FINAL_FULL_RUN = 'student_v5_full'\n"
        "KAGGLE_PUBLIC = {\n"
        "    'wide3': 0.875,\n"
        "    'student_full': 0.887,\n"
        "    'student_long_full': 0.896,\n"
        "    'student_v5_full': 0.899,\n"
        "}\n\n"
        "def read_json(path):\n"
        "    path = Path(path)\n"
        "    return json.loads(path.read_text()) if path.exists() else None\n\n"
        "plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3})"
    ),
    md("## 2. Data"),
    code(
        "train_split = load_split('data', 'train', '.cache/data')\n"
        "test_split = load_split('data', 'test', '.cache/data')\n"
        "print('train', train_split.images.shape, '| test', test_split.images.shape)\n"
        "print(pd.Series(train_split.labels).map(dict(enumerate(CLASSES))).value_counts().sort_index())\n\n"
        "fig, axes = plt.subplots(2, 10, figsize=(14, 3))\n"
        "for c, col in enumerate(axes.T):\n"
        "    idx = np.flatnonzero(train_split.labels == c)[:2]\n"
        "    for ax, i in zip(col, idx):\n"
        "        ax.imshow(train_split.images[i]); ax.axis('off')\n"
        "    col[0].set_title(CLASSES[c], fontsize=9)\n"
        "plt.suptitle('Training images'); plt.tight_layout(); plt.show()\n\n"
        "fig, axes = plt.subplots(1, 10, figsize=(14, 1.8))\n"
        "for ax, img in zip(axes, test_split.images[:10]):\n"
        "    ax.imshow(img); ax.axis('off')\n"
        "plt.suptitle('Test images (unlabelled)'); plt.tight_layout(); plt.show()"
    ),
    md("## 3. Model and parameter count"),
    code(
        "cfg = load_config(FINAL_CONFIG)\n"
        "model = build_model(cfg.model)\n"
        "print(model)\n"
        "print(layer_table(model))\n"
        "N_PARAMS = count_parameters(model)\n"
        "assert N_PARAMS <= 5_000_000\n"
        "print(f'Trainable parameters: {N_PARAMS:,}  (budget 5,000,000)')"
    ),
    md("## 4. All runs"),
    code(
        "rows = [json.loads(p.read_text()) for p in sorted(Path('runs').glob('*/summary.json'))]\n"
        "runs = pd.DataFrame(rows).set_index('run_name')\n"
        "runs['kaggle_public'] = pd.Series(KAGGLE_PUBLIC)\n"
        "cols = ['params', 'epochs', 'train_images', 'best_val_acc', 'best_val_acc_tta', 'kaggle_public', 'train_time_min']\n"
        "runs = runs[[c for c in cols if c in runs]]\n"
        "runs.sort_values('best_val_acc', ascending=False)"
    ),
    md(
        "## 5. Ablations (60 epochs each)\n\n"
        "Same recipe, one change at a time, on the 45k/5k split. Short runs rank the choices; "
        "the winners were then trained for 200+ epochs."
    ),
    code(
        "abl = runs[runs.index.str.startswith('abl_')].copy()\n"
        "abl.index = abl.index.str.replace('abl_', '')\n"
        "abl = abl.sort_values('best_val_acc')\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.barh(abl.index, abl.best_val_acc * 100)\n"
        "for y, v in enumerate(abl.best_val_acc * 100):\n"
        "    ax.text(v + 0.05, y, f'{v:.2f}', va='center', fontsize=8)\n"
        "ax.set(xlim=(abl.best_val_acc.min() * 100 - 1, abl.best_val_acc.max() * 100 + 1),\n"
        "       xlabel='validation accuracy (%)', title='Ablations at 60 epochs')\n"
        "plt.tight_layout(); plt.show()\n"
        "abl[['params', 'best_val_acc', 'best_val_acc_tta', 'train_time_min']]"
    ),
    md("## 6. Training curves"),
    code(
        "CURVES = ['wide3', 'student', 'student_long', 'student_v5', 'teacher', 'teacher_s3']\n"
        "fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))\n"
        "for name in CURVES:\n"
        "    path = Path('runs') / name / 'metrics.csv'\n"
        "    if not path.exists():\n"
        "        continue\n"
        "    m = pd.read_csv(path)\n"
        "    acc = m.ema_val_acc.where(m.ema_val_acc.notna(), m.val_acc) if 'ema_val_acc' in m else m.val_acc\n"
        "    ax[0].plot(m.epoch, acc * 100, label=name)\n"
        "    ax[1].plot(m.epoch / m.epoch.max(), acc * 100, label=name)\n"
        "ax[0].set(title='Validation accuracy (EMA weights)', xlabel='epoch', ylim=(90, 98))\n"
        "ax[1].set(title='Same, by fraction of the schedule', xlabel='fraction of training', ylim=(90, 98))\n"
        "for a in ax:\n"
        "    a.legend()\n"
        "plt.tight_layout(); plt.show()\n\n"
        "m = pd.read_csv(Path('runs') / FINAL_RUN / 'metrics.csv')\n"
        "fig, ax = plt.subplots(1, 3, figsize=(16, 4))\n"
        "ax[0].plot(m.epoch, m.train_loss, label='train (KD + CE)'); ax[0].plot(m.epoch, m.val_loss, label='val CE')\n"
        "ax[0].set(title=f'{FINAL_RUN}: loss', xlabel='epoch'); ax[0].legend()\n"
        "ax[1].plot(m.epoch, m.train_acc, label='train (augmented, CutMix)')\n"
        "ax[1].plot(m.epoch, m.val_acc, label='val')\n"
        "ax[1].plot(m.epoch, m.ema_val_acc, label='val (EMA weights)')\n"
        "ax[1].set(title=f'{FINAL_RUN}: accuracy', xlabel='epoch', ylim=(0.5, 1)); ax[1].legend()\n"
        "ax[2].plot(m.epoch, m.lr); ax[2].set(title='Learning rate', xlabel='epoch')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 7. Teachers, distillation and schedule length"),
    code(
        "ladder = ['wide3', 'student', 'student_long', 'student_v5', 'teacher', 'teacher_s1', 'teacher_s2', 'teacher_s3', 'teacher_s4']\n"
        "ladder = [r for r in ladder if r in runs.index]\n"
        "sub = runs.loc[ladder, ['params', 'epochs', 'best_val_acc', 'best_val_acc_tta']]\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "colors = ['tab:red' if p > 5_000_000 else 'tab:blue' for p in sub.params]\n"
        "ax.bar(sub.index, sub.best_val_acc * 100, color=colors)\n"
        "for x, v in enumerate(sub.best_val_acc * 100):\n"
        "    ax.text(x, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)\n"
        "ax.set(ylim=(95.5, 98), ylabel='validation accuracy (%)',\n"
        "       title='Blue: submittable (<5M params). Red: teachers (36.7M), used only for distillation')\n"
        "ax.tick_params(axis='x', rotation=30)\n"
        "plt.tight_layout(); plt.show()\n"
        "sub"
    ),
    md("## 8. Validation analysis of the final model"),
    code(
        "EVAL_RUN = next(r for r in [FINAL_RUN, 'student_long'] if (Path('runs') / r / 'val_eval.json').exists())\n"
        "ev = read_json(Path('runs') / EVAL_RUN / 'val_eval.json')\n"
        "print(f'run {EVAL_RUN}:', {k: round(v, 4) for k, v in ev['accuracy'].items()}, f\"({ev['val_size']} held-out images)\")\n"
        "cm = np.array(ev['confusion'])\n"
        "fig, ax = plt.subplots(1, 2, figsize=(14, 5))\n"
        "ax[0].bar(CLASSES, cm.diagonal() / cm.sum(1) * 100); ax[0].set(title=f'{EVAL_RUN}: per-class accuracy (%)', ylim=(90, 100))\n"
        "ax[0].tick_params(axis='x', rotation=45)\n"
        "ax[1].imshow(cm, cmap='Blues'); ax[1].set(title=f\"{EVAL_RUN}: confusion matrix (tta={ev['confusion_tta']})\", xlabel='predicted', ylabel='true')\n"
        "ax[1].set_xticks(range(10), CLASSES, rotation=90); ax[1].set_yticks(range(10), CLASSES)\n"
        "ax[1].grid(False)\n"
        "for i in range(10):\n"
        "    for j in range(10):\n"
        "        ax[1].text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='white' if cm[i, j] > cm.max() / 2 else 'black')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 9. Final model and Kaggle submission\n\n"
        "The submitted model is the same config retrained on all 50k images "
        "(`data.val_size=0`), predicted with flip TTA by `cifar-predict`."
    ),
    code(
        "sub = pd.read_csv(f'submissions/{FINAL_FULL_RUN}.csv')\n"
        "sample = pd.read_csv('data/sample_submission.csv')\n"
        "assert list(sub.columns) == ['id', 'label'] and (sub.id.values == sample.id.values).all()\n"
        "print(sub.label.map(dict(enumerate(CLASSES))).value_counts().sort_index())\n\n"
        "final = runs.loc[FINAL_FULL_RUN]\n"
        "val = runs.loc[FINAL_RUN]\n"
        "print('=' * 64)\n"
        "print(f'Final model            : {FINAL_FULL_RUN} ({FINAL_CONFIG}, {int(final.epochs)} epochs, {int(final.train_images)} images)')\n"
        "print(f'Trainable parameters   : {int(final.params):,} (limit 5,000,000)')\n"
        "print(f'Validation accuracy    : {val.best_val_acc:.4f} plain, {val.best_val_acc_tta:.4f} with flip TTA (5k held-out, run {FINAL_RUN})')\n"
        "print(f'Kaggle public accuracy : {KAGGLE_PUBLIC.get(FINAL_FULL_RUN)}')\n"
        "print('=' * 64)"
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("project1.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
