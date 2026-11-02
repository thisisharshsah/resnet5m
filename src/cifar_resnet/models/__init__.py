from __future__ import annotations

from torch import nn

from cifar_resnet.config import ModelConfig
from cifar_resnet.models.ensemble import LogitEnsemble
from cifar_resnet.models.resnet import BasicBlock, ResNet

__all__ = ["BasicBlock", "LogitEnsemble", "ResNet", "build_model", "count_parameters"]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def build_model(cfg: ModelConfig, enforce_budget: bool = True) -> ResNet:
    model = ResNet(
        channels=cfg.channels,
        blocks=cfg.blocks,
        kernel_size=cfg.kernel_size,
        skip_kernel_size=cfg.skip_kernel_size,
        stem_channels=cfg.stem_channels,
        pool_size=cfg.pool_size,
        dropout=cfg.dropout,
        se_ratio=cfg.se_ratio,
        zero_init_residual=cfg.zero_init_residual,
        num_classes=cfg.num_classes,
    )
    n_params = count_parameters(model)
    if enforce_budget and n_params > cfg.max_params:
        raise ValueError(
            f"Model has {n_params:,} trainable parameters, over the {cfg.max_params:,} budget"
        )
    return model
