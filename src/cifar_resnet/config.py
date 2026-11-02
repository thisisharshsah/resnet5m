from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data"
    cache_dir: str = ".cache/data"
    val_size: int = 5000
    split_seed: int = 42
    num_workers: int = 8


@dataclass
class AugmentConfig:
    policy: str = "autoaugment"
    crop_padding: int = 4
    hflip: bool = True
    erasing_prob: float = 0.25
    mix: str = "none"
    mix_alpha: float = 1.0
    mix_prob: float = 0.5


@dataclass
class ModelConfig:
    channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    blocks: list[int] = field(default_factory=lambda: [2, 2, 2])
    kernel_size: int = 3
    skip_kernel_size: int = 1
    stem_channels: int | None = None
    pool_size: int | None = None
    dropout: float = 0.0
    se_ratio: float = 0.0
    zero_init_residual: bool = True
    num_classes: int = 10
    max_params: int = 5_000_000


@dataclass
class OptimConfig:
    name: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    scheduler: str = "cosine"
    warmup_epochs: float = 5.0
    min_lr_ratio: float = 0.0
    sam_rho: float = 0.0
    sam_adaptive: bool = False


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    amp: bool = True
    channels_last: bool = True
    ema_decay: float = 0.999
    seed: int = 42
    eval_every: int = 1
    max_steps_per_epoch: int | None = None
    tta: bool = True
    auto_resume: bool = False


@dataclass
class DistillConfig:
    teacher: str | list[str] | None = None

    alpha: float = 0.9
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
    cfg = Config()
    if path is not None:
        with open(path) as f:
            _apply(cfg, yaml.safe_load(f) or {})
    return apply_overrides(cfg, overrides)
