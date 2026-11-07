import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
OUT = HERE / "figures"

WIDTH = 6.5
DPI = 220
BLUE, GREY, BAND, INK, EDGE, LIGHT = (
    "#2563a8",
    "#b8b8b8",
    "#ececec",
    "#222222",
    "#6b6b6b",
    "#e4eef9",
)

FINAL_RUN = "student_v5"
ABLATION_BASELINE = "abl_arch_wide3"
ABLATIONS = {
    "abl_arch_wide3_wider": "Wider: 96, 192, 320 channels",
    "abl_no_ema": "No weight EMA",
    "abl_sam": "Add SAM",
    "abl_arch_deep3": "Three blocks per stage",
    "abl_arch_wide3_skip3": "3x3 shortcut convolutions",
    "abl_aug_cutmix": "Add CutMix",
    "abl_aug_trivialaugment": "TrivialAugment instead of AutoAugment",
    "abl_no_label_smoothing": "No label smoothing",
    "abl_aug_basic": "No AutoAugment, no random erasing",
    "abl_arch_resnet18_narrow": "ResNet-18 layout (4 narrow stages)",
}
NOISE_BAND = 0.6
TOP_MISTAKES = 8


def best_val(run: str) -> float:
    summary = json.loads((RUNS / run / "summary.json").read_text())
    return round(summary["best_val_acc"] * 100, 2)


def canvas(height: float):
    fig = plt.figure(figsize=(WIDTH, height), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, face="white", edge=EDGE, lw=1.0, **style):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.05",
            fc=face,
            ec=edge,
            lw=lw,
            **style,
        )
    )


def arrow(ax, start, end, scale=9, lw=1.0):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=INK)
    )


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)


def pipeline(height: float = 1.05) -> None:
    fig, ax = canvas(height)
    steps = [
        ("Training set", ["50,000 images", "32x32, 10 classes"], 1.02, False),
        ("Split", ["45,000 train", "5,000 validation"], 1.02, False),
        ("5 teachers", ["36.7M params each", "from scratch"], 1.08, False),
        ("Student", ["4.9M params, learns", "from labels and the", "teachers' average"], 1.3, True),
        ("Final model", ["retrained on all", "50,000 images", "Kaggle: 0.899"], 1.08, False),
    ]
    gap = (WIDTH - 0.04 - sum(step[2] for step in steps)) / (len(steps) - 1)
    y, bh, x = 0.06, height - 0.12, 0.02
    for i, (title, lines, w, highlight) in enumerate(steps):
        box(
            ax,
            x,
            y,
            w,
            bh,
            face=LIGHT if highlight else "white",
            edge=BLUE if highlight else EDGE,
            lw=1.3 if highlight else 1.0,
        )
        top = y + bh / 2 + (0.17 + 0.155 * len(lines)) / 2
        ax.text(
            x + w / 2,
            top,
            title,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=BLUE if highlight else INK,
        )
        for k, line in enumerate(lines):
            ax.text(
                x + w / 2,
                top - 0.19 - 0.155 * k,
                line,
                ha="center",
                va="top",
                fontsize=8.5,
                color=INK,
            )
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.02, y + bh / 2), (x + w + gap - 0.02, y + bh / 2))
        x += w + gap
    save(fig, "pipeline.png")


def architecture(height: float = 1.68) -> None:
    fig, ax = canvas(height)
    parts = [
        ("Input", ["3 channels", "32x32"], 3, 0.78),
        ("Stem", ["conv 3x3", "96 ch, 32x32"], 96, 0.86),
        ("Stage 1", ["2 blocks, 96 ch", "32x32, 7% of params"], 96, 1.12),
        ("Stage 2", ["2 blocks, 192 ch", "16x16, 24% of params"], 192, 1.12),
        ("Stage 3", ["2 blocks, 320 ch", "8x8, 69% of params"], 320, 1.12),
        ("Output", ["avg pool + linear", "10 classes"], 10, 1.0),
    ]
    gap = (WIDTH - 0.04 - sum(part[3] for part in parts)) / (len(parts) - 1)
    y0 = 0.4
    scale = (height - 0.05 - y0) / 320
    x, lefts = 0.02, []
    for i, (name, lines, channels, w) in enumerate(parts):
        lefts.append(x)
        stage = name.startswith("Stage")
        box(
            ax,
            x,
            y0,
            w,
            max(0.24, channels * scale),
            face=LIGHT if stage else "white",
            edge=BLUE if stage else EDGE,
            lw=1.2 if stage else 1.0,
        )
        ax.text(
            x + w / 2,
            y0 + 0.12,
            name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=BLUE if stage else INK,
        )
        for k, line in enumerate(lines):
            ax.text(
                x + w / 2, y0 - 0.06 - 0.15 * k, line, ha="center", va="top", fontsize=8, color=INK
            )
        if i < len(parts) - 1:
            arrow(ax, (x + w + 0.02, y0 + 0.12), (x + w + gap - 0.02, y0 + 0.12))
        x += w + gap

    ix0, ix1 = 0.02, lefts[3] - 0.12
    iy0, iy1 = y0 + 96 * scale + 0.07, height - 0.03
    box(ax, ix0, iy0, ix1 - ix0, iy1 - iy0, face="#fafafa", lw=0.8, ls=(0, (3, 2)))
    ax.text(
        ix0 + 0.08,
        iy1 - 0.05,
        "One residual block (each stage has two)",
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )
    ym, bh = iy0 + 0.22, 0.3
    ax.text(ix0 + 0.12, ym, "x", ha="center", va="center", fontsize=9.5, style="italic")
    arrow(ax, (ix0 + 0.18, ym), (ix0 + 0.34, ym))
    for a, b, label in ((0.36, 1.0, "conv 3x3\nBN, ReLU"), (1.2, 1.84, "conv 3x3\nBN")):
        box(ax, ix0 + a, ym - bh / 2, b - a, bh, lw=0.9)
        ax.text(
            ix0 + (a + b) / 2, ym, label, ha="center", va="center", fontsize=7.5, linespacing=1.1
        )
    cx = ix0 + 2.06
    arrow(ax, (ix0 + 1.02, ym), (ix0 + 1.18, ym))
    arrow(ax, (ix0 + 1.86, ym), (cx - 0.09, ym))
    ax.add_patch(Circle((cx, ym), 0.075, fc="white", ec=INK, lw=0.9))
    ax.text(cx, ym - 0.005, "+", ha="center", va="center", fontsize=9)
    arrow(ax, (cx + 0.08, ym), (ix0 + 2.28, ym))
    box(ax, ix0 + 2.3, ym - 0.11, 0.36, 0.22, lw=0.9)
    ax.text(ix0 + 2.48, ym, "ReLU", ha="center", va="center", fontsize=7.5)
    arrow(ax, (ix0 + 2.68, ym), (ix0 + 2.8, ym))
    ax.text(ix0 + 2.86, ym, "y", ha="center", va="center", fontsize=9.5, style="italic")
    yl = ym + bh / 2 + 0.1
    ax.plot([ix0 + 0.24, ix0 + 0.24, cx, cx], [ym + 0.05, yl, yl, ym + 0.09], color=INK, lw=0.9)
    arrow(ax, (cx, yl), (cx, ym + 0.08), scale=8, lw=0.9)
    ax.text(
        (ix0 + 0.24 + cx) / 2,
        yl + 0.025,
        "shortcut: identity, or 1x1 conv + BN",
        ha="center",
        va="bottom",
        fontsize=7.5,
        style="italic",
        color="#333",
    )
    save(fig, "architecture.png")


def ablations(height: float = 3.0) -> None:
    base = best_val(ABLATION_BASELINE)
    rows = [(label, round(best_val(run) - base, 2)) for run, label in ABLATIONS.items()]
    rows.sort(key=lambda row: -row[1])
    labels = [label for label, _ in rows][::-1]
    deltas = [delta for _, delta in rows][::-1]
    fig, ax = plt.subplots(figsize=(WIDTH, height), dpi=DPI)
    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=BAND, zorder=0)
    colors = [BLUE if abs(d) > NOISE_BAND else GREY for d in deltas]
    ax.barh(range(len(rows)), deltas, color=colors, height=0.6, zorder=2)
    for i, d in enumerate(deltas):
        ax.text(d - 0.03, i, f"{d:+.2f}", va="center", ha="right", fontsize=9)
    ax.axvline(0, color="#444", lw=0.8, zorder=3)
    ax.set_yticks(range(len(rows)), labels, fontsize=9.5)
    ax.set_xlim(-2.3, 0.75)
    ax.set_xlabel(
        f"Change in validation accuracy (points) vs. baseline at {base:.2f}%", fontsize=9.5
    )
    ax.text(0.02, len(rows) - 0.35, "noise band\n(about 2 SE)", fontsize=8, color="#555", va="top")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "ablations.png")


def per_class(height: float = 2.6) -> None:
    ev = json.loads((RUNS / FINAL_RUN / "val_eval.json").read_text())
    classes = ev["classes"]
    confusion = np.array(ev["confusion"])
    accuracy = confusion.diagonal() / confusion.sum(axis=1) * 100
    overall = confusion.trace() / confusion.sum() * 100
    order = sorted(range(len(classes)), key=lambda i: (accuracy[i], classes[i]))
    mistakes = sorted(
        (
            (int(confusion[i, j]), classes[i], classes[j])
            for i in range(len(classes))
            for j in range(len(classes))
            if i != j
        ),
        reverse=True,
    )[:TOP_MISTAKES]

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(WIDTH, height),
        dpi=DPI,
        gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55},
    )
    rows = range(len(order))
    left.axvline(overall, color="#444", ls="--", lw=0.8, zorder=1)
    left.scatter(
        [accuracy[i] for i in order],
        rows,
        s=42,
        zorder=3,
        color=[BLUE if r < 2 else GREY for r in rows],
    )
    for r, i in zip(rows, order):
        left.text(accuracy[i] + 0.25, r, f"{accuracy[i]:.1f}", va="center", ha="left", fontsize=8.5)
    left.set_yticks(list(rows), [classes[i] for i in order], fontsize=9)
    left.set_xlim(93, 100.4)
    left.set_ylim(len(order) - 0.4, -0.6)
    left.text(
        overall - 0.12,
        len(order) - 0.55,
        f"all {overall:.2f}%",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#444",
    )
    left.set_xlabel("Accuracy per class (%)", fontsize=9)
    left.grid(axis="x", color=BAND, zorder=0)

    rows = range(len(mistakes))
    counts = [count for count, _, _ in mistakes]
    right.barh(
        list(rows), counts, height=0.62, zorder=2, color=[BLUE if r < 2 else GREY for r in rows]
    )
    for r, count in zip(rows, counts):
        right.text(count + 0.3, r, str(count), va="center", ha="left", fontsize=8.5)
    right.set_yticks(list(rows), [f"{a} → {b}" for _, a, b in mistakes], fontsize=9)
    right.set_ylim(len(mistakes) - 0.4, -0.6)
    right.set_xlim(0, max(counts) + 2.5)
    right.set_xlabel("Images (true class → predicted class)", fontsize=9)
    right.grid(axis="x", color=BAND, zorder=0)

    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.2)
    save(fig, "per_class.png")


def main() -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    OUT.mkdir(exist_ok=True)
    pipeline()
    architecture()
    ablations()
    per_class()


if __name__ == "__main__":
    main()
