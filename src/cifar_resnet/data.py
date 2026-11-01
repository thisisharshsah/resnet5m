"""Data loading for the Kaggle competition layout.

Expected layout (exactly what Kaggle's "Download All" zip contains)::

    <root>/
        train_labels.csv    # id, filename, label
        test.csv            # id, filename
        sample_submission.csv
        train_images/*.png
        test_images/*.png

PNGs are decoded once and cached as uint8 arrays, so later runs start in seconds.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from cifar_resnet.config import AugmentConfig, Config

log = logging.getLogger(__name__)

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
# Standard CIFAR-10 training-set statistics.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

_SPLITS = {
    "train": ("train_labels.csv", "train_images"),
    "test": ("test.csv", "test_images"),
}


@dataclass
class Split:
    images: np.ndarray  # (N, 32, 32, 3) uint8
    ids: np.ndarray  # (N,) int64, as given in the CSV
    labels: np.ndarray | None  # (N,) int64, None for the test split


def find_data_root(root: str | Path) -> Path:
    """Return the folder containing ``train_labels.csv``, searching one level down."""
    root = Path(root).expanduser()
    if (root / "train_labels.csv").exists():
        return root
    for child in sorted(p for p in root.glob("*") if p.is_dir()):
        if (child / "train_labels.csv").exists():
            return child
    raise FileNotFoundError(
        f"Could not find train_labels.csv in {root} or its subfolders. "
        "Download the competition data from Kaggle and unzip it there."
    )


def _resolve_image_path(root: Path, image_dir: str, filename: str) -> Path:
    for candidate in (root / image_dir / filename, root / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image {filename} not found under {root / image_dir}")


def _read_png(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if arr.shape != (32, 32, 3):
        raise ValueError(f"{path} has shape {arr.shape}, expected (32, 32, 3)")
    return arr


def load_split(root: str | Path, split: str, cache_dir: str | Path | None = None) -> Split:
    """Load ``train`` or ``test`` images as uint8 arrays, using an on-disk cache."""
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {list(_SPLITS)}")
    root = find_data_root(root)
    csv_name, image_dir = _SPLITS[split]
    df = pd.read_csv(root / csv_name)
    ids = df["id"].to_numpy(dtype=np.int64)
    labels = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None

    cache_file = None
    if cache_dir is not None:
        # Key the cache on the CSV contents so a changed dataset never reuses stale arrays.
        digest = hashlib.md5((root / csv_name).read_bytes()).hexdigest()[:10]
        cache_file = Path(cache_dir) / f"{split}-{digest}.npy"
        if cache_file.exists():
            images = np.load(cache_file)
            if len(images) == len(df):
                return Split(images=images, ids=ids, labels=labels)

    first = _resolve_image_path(root, image_dir, df["filename"].iloc[0])
    base = first.parent
    paths = [base / name for name in df["filename"]]
    log.info("Decoding %d %s images from %s (one-time)", len(paths), split, base)
    with ThreadPoolExecutor(max_workers=16) as pool:
        images = np.stack(list(pool.map(_read_png, paths)))

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, images)
    return Split(images=images, ids=ids, labels=labels)


def stratified_split(labels: np.ndarray, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train/val with the same class balance in both."""
    if val_size <= 0:
        return np.arange(len(labels)), np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = val_size // len(classes)
    val_idx = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        val_idx.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    val = np.sort(np.concatenate(val_idx))
    train = np.setdiff1d(np.arange(len(labels)), val)
    return train, val


def build_transforms(aug: AugmentConfig | None, train: bool) -> v2.Compose:
    """Transforms operate on uint8 CHW tensors."""
    ops: list[v2.Transform] = []
    if train and aug is not None:
        if aug.crop_padding > 0:
            ops.append(v2.RandomCrop(32, padding=aug.crop_padding, padding_mode="reflect"))
        if aug.hflip:
            ops.append(v2.RandomHorizontalFlip())
        policy = aug.policy.lower()
        if policy == "autoaugment":
            ops.append(v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))
        elif policy == "trivialaugment":
            ops.append(v2.TrivialAugmentWide())
        elif policy == "randaugment":
            ops.append(v2.RandAugment())
        elif policy != "none":
            raise ValueError(f"Unknown augmentation policy {aug.policy!r}")
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    if train and aug is not None and aug.erasing_prob > 0:
        ops.append(v2.RandomErasing(p=aug.erasing_prob, value=0))
    return v2.Compose(ops)


class ArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray | None = None,
        transform: v2.Compose | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img = torch.from_numpy(self.images[index]).permute(2, 0, 1).contiguous()
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.labels[index]) if self.labels is not None else -1
        return img, label


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=torch.device(device).type == "cuda",
        persistent_workers=workers > 0,
    )


def build_train_val_loaders(
    cfg: Config, device: torch.device | str = "cpu"
) -> tuple[DataLoader, DataLoader | None]:
    split = load_split(cfg.data.root, "train", cfg.data.cache_dir)
    assert split.labels is not None
    train_idx, val_idx = stratified_split(split.labels, cfg.data.val_size, cfg.train.seed)
    log.info("Training images: %d | validation images: %d", len(train_idx), len(val_idx))

    train_ds = ArrayDataset(
        split.images[train_idx], split.labels[train_idx], build_transforms(cfg.augment, True)
    )
    train_loader = _loader(train_ds, cfg.train.batch_size, True, cfg.data.num_workers, device)
    if len(val_idx) == 0:
        return train_loader, None
    val_ds = ArrayDataset(
        split.images[val_idx], split.labels[val_idx], build_transforms(None, False)
    )
    val_loader = _loader(val_ds, 512, False, cfg.data.num_workers, device)
    return train_loader, val_loader


def build_val_split(cfg: Config) -> ArrayDataset:
    """The exact validation subset a run held out (same seed and size)."""
    split = load_split(cfg.data.root, "train", cfg.data.cache_dir)
    assert split.labels is not None
    _, val_idx = stratified_split(split.labels, cfg.data.val_size, cfg.train.seed)
    return ArrayDataset(split.images[val_idx], split.labels[val_idx], build_transforms(None, False))


def build_test_loader(
    root: str | Path,
    cache_dir: str | Path | None,
    batch_size: int = 512,
    workers: int = 4,
    device: torch.device | str = "cpu",
) -> tuple[DataLoader, np.ndarray]:
    split = load_split(root, "test", cache_dir)
    ds = ArrayDataset(split.images, None, build_transforms(None, False))
    return _loader(ds, batch_size, False, workers, device), split.ids
