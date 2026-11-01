"""Typed experiment configuration.

Every experiment is a YAML file under ``configs/`` that overrides the defaults below.
Any value can also be overridden from the command line with dotted keys, e.g.::

    cifar-train --config configs/wrn16_5.yaml train.epochs=50 data.val_size=0
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data"  # folder holding train_labels.csv, test.csv, train_images/, test_images/
    cache_dir: str = ".cache/data"  # decoded uint8 arrays are cached here after the first run
    val_size: int = 5000  # stratified hold-out from the training set; 0 trains on all 50k
    num_workers: int = 8


@dataclass
class AugmentConfig:
    policy: str = "autoaugment"  # none | autoaugment | trivialaugment | randaugment
    crop_padding: int = 4
    hflip: bool = True
    erasing_prob: float = 0.25  # RandomErasing (Cutout-style), applied after normalization
    mix: str = "none"  # none | mixup | cutmix
    mix_alpha: float = 1.0
    mix_prob: float = 0.5  # fraction of batches that get mixed


@dataclass
class ModelConfig:
    channels: list[int] = field(default_factory=lambda: [64, 128, 256])  # C_i per stage
    blocks: list[int] = field(default_factory=lambda: [2, 2, 2])  # residual blocks per stage
    kernel_size: int = 3  # F_i, conv filter size inside residual blocks
    skip_kernel_size: int = 1  # K_i, kernel size of projection shortcuts
    stem_channels: int | None = None  # defaults to channels[0]
    pool_size: int | None = None  # P; None means global average pooling
    dropout: float = 0.0
    se_ratio: float = 0.0  # squeeze-and-excitation reduction (0 disables)
    zero_init_residual: bool = True
    num_classes: int = 10
    max_params: int = 5_000_000  # hard project constraint, enforced at build time


@dataclass
class OptimConfig:
    name: str = "sgd"  # sgd | adamw
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    scheduler: str = "cosine"  # cosine | onecycle | constant
    warmup_epochs: float = 5.0
    min_lr_ratio: float = 0.0  # final lr = lr * min_lr_ratio (cosine only)
    sam_rho: float = 0.0  # Sharpness-Aware Minimization radius (0 disables; 0.05 typical)
    sam_adaptive: bool = False  # adaptive SAM (ASAM); use a larger rho, e.g. 1.0-2.0


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    amp: bool = True  # mixed precision on CUDA (bf16 when supported, else fp16)
    channels_last: bool = True  # CUDA only
    ema_decay: float = 0.999  # 0 disables the weight EMA
    seed: int = 42
    eval_every: int = 1
    max_steps_per_epoch: int | None = None  # for smoke tests only
    tta: bool = True  # horizontal-flip test-time augmentation when evaluating
    auto_resume: bool = False  # continue from runs/<run_name>/last.pt if present (HPC requeue)


@dataclass
class DistillConfig:
    teacher: str | None = None  # checkpoint of our own from-scratch teacher (see rules.py)
    alpha: float = 0.9  # weight of the distillation term (1 - alpha on the hard labels)
    temperature: float = 4.0


@dataclass
class Config:
    run_name: str = "default"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _apply(target: Any, updates: dict[str, Any], prefix: str = "") -> None:
    for key, value in updates.items():
        if not hasattr(target, key):
            raise KeyError(f"Unknown config key: {prefix}{key}")
        current = getattr(target, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be a mapping, got {value!r}")
            _apply(current, value, f"{prefix}{key}.")
        else:
            setattr(target, key, value)


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=[1,2]"]`` into a nested dict, parsing values as YAML."""
    nested: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = nested
        *parents, leaf = dotted.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return nested


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    _apply(cfg, parse_overrides(overrides))
    return cfg


def from_dict(values: dict[str, Any]) -> Config:
    cfg = Config()
    _apply(cfg, values)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load defaults, then the YAML file, then command-line overrides (later wins)."""
    cfg = Config()
    if path is not None:
        with open(path) as f:
            _apply(cfg, yaml.safe_load(f) or {})
    return apply_overrides(cfg, overrides)
