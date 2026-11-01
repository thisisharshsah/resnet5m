"""Saving and restoring checkpoints. A checkpoint carries its own config, so a model
can always be rebuilt from the file alone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from cifar_resnet.config import Config, from_dict
from cifar_resnet.models import build_model, count_parameters
from cifar_resnet.rules import CHECKPOINT_ORIGIN


def save_weights(path: str | Path, model: nn.Module, cfg: Config, **meta: Any) -> None:
    """Small inference checkpoint: weights + config + metadata, stamped with its origin."""
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "params": count_parameters(model),
            "origin": CHECKPOINT_ORIGIN,
            **meta,
        },
        path,
    )


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(
    path: str | Path, device: torch.device | str = "cpu", enforce_budget: bool = True
) -> tuple[nn.Module, Config, dict]:
    """Rebuild the model stored in ``path`` and load its weights."""
    ckpt = read_checkpoint(path)
    cfg = from_dict(ckpt["config"])
    model = build_model(cfg.model, enforce_budget=enforce_budget)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt
