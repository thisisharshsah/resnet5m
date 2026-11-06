from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cifar_resnet.config import AugmentConfig, DistillConfig, OptimConfig
from cifar_resnet.ema import ModelEMA
from cifar_resnet.sam import SAM, frozen_bn_stats
from cifar_resnet.utils import autocast_context


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    name = cfg.name.lower()
    if name == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    raise ValueError(f"Unknown optimizer {cfg.name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: OptimConfig, steps_per_epoch: int, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    total = max(1, steps_per_epoch * epochs)
    warmup = int(cfg.warmup_epochs * steps_per_epoch)
    kind = cfg.scheduler.lower()

    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=total, pct_start=0.15, anneal_strategy="linear"
        )

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        if kind != "cosine":
            raise ValueError(f"Unknown scheduler {cfg.scheduler!r}")
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def mix_batch(
    x: torch.Tensor, y: torch.Tensor, aug: AugmentConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    mode = aug.mix.lower()
    if mode == "none" or np.random.rand() >= aug.mix_prob:
        return x, y, y, 1.0
    lam = float(np.random.beta(aug.mix_alpha, aug.mix_alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], y, y[perm], lam
    if mode == "cutmix":
        h, w = x.shape[-2:]
        cut = math.sqrt(1 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y1 = np.clip([cy - ch // 2, cy + ch // 2], 0, h)
        x0, x1 = np.clip([cx - cw // 2, cx + cw // 2], 0, w)
        x = x.clone()
        x[..., y0:y1, x0:x1] = x[perm][..., y0:y1, x0:x1]
        lam = 1 - ((y1 - y0) * (x1 - x0) / (h * w))
        return x, y, y[perm], float(lam)
    raise ValueError(f"Unknown mix mode {aug.mix!r}")


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, t: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student / t, dim=1), F.softmax(teacher / t, dim=1), reduction="batchmean"
    ) * (t * t)


@dataclass
class EpochStats:
    loss: float
    acc: float


def _forward_loss(
    model: nn.Module,
    batch: tuple,
    criterion: nn.Module,
    distill: DistillConfig | None,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y_a, y_b, lam, teacher_logits = batch
    with autocast_context(device, amp):
        out = model(x)
        loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
        if teacher_logits is not None and distill is not None:
            kd = distillation_loss(out.float(), teacher_logits, distill.temperature)
            loss = (1 - distill.alpha) * loss + distill.alpha * kd
    return out, loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    aug: AugmentConfig,
    amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEMA | None = None,
    teacher: nn.Module | None = None,
    distill: DistillConfig | None = None,
    channels_last: bool = False,
    max_steps: int | None = None,
    sam: SAM | None = None,
) -> EpochStats:
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss, correct, seen = 0.0, 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x, y_a, y_b, lam = mix_batch(x, y, aug)

        teacher_logits = None
        if teacher is not None and distill is not None:
            with torch.no_grad(), autocast_context(device, amp):
                teacher_logits = teacher(x).float()

        batch = (x, y_a, y_b, lam, teacher_logits)
        logits, loss = _forward_loss(model, batch, criterion, distill, device, amp)
        optimizer.zero_grad(set_to_none=True)
        if sam is not None:
            loss.backward()
            sam.first_step()
            optimizer.zero_grad(set_to_none=True)
            with frozen_bn_stats(model):
                _forward_loss(model, batch, criterion, distill, device, amp)[1].backward()
            sam.second_step()
        elif scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        with torch.no_grad():
            pred = logits.argmax(1)
            correct += (
                lam * (pred == y_a).float().sum() + (1 - lam) * (pred == y_b).float().sum()
            ).item()
        total_loss += loss.item() * y.size(0)
        seen += y.size(0)
    return EpochStats(loss=total_loss / max(1, seen), acc=correct / max(1, seen))


TTA_MODES = ("none", "flip", "shift")


def tta_views(x: torch.Tensor, tta: bool | str) -> list[torch.Tensor]:
    mode = {True: "flip", False: "none"}.get(tta, tta)
    if mode not in TTA_MODES:
        raise ValueError(f"tta must be one of {TTA_MODES}, got {tta!r}")
    if mode == "none":
        return [x]
    views = [x, torch.flip(x, dims=[3])]
    if mode == "shift":
        size = x.shape[-1]
        padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
        for dy, dx in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            crop = padded[:, :, 2 + dy : 2 + dy + size, 2 + dx : 2 + dx + size]
            views += [crop, torch.flip(crop, dims=[3])]
    return views


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool | str = False,
    amp: bool = True,
    channels_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with autocast_context(device, amp):
            logits = torch.stack([model(v).float() for v in tta_views(x, tta)]).mean(0)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool | str = False,
    amp: bool = True,
    channels_last: bool = False,
) -> EpochStats:
    logits, labels = predict_logits(model, loader, device, tta, amp, channels_last)
    loss = F.cross_entropy(logits, labels).item()
    acc = (logits.argmax(1) == labels).float().mean().item()
    return EpochStats(loss=loss, acc=acc)
