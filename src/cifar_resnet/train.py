"""Train a ResNet on CIFAR-10.

Usage::

    cifar-train --config configs/student.yaml
    cifar-train --config configs/student.yaml data.val_size=0 run_name=student_full
    cifar-train --resume runs/student/last.pt

Each run writes to ``runs/<run_name>/``: ``config.yaml``, ``train.log``,
``metrics.csv`` (one row per epoch), ``best.pt`` / ``last.pt`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch
from torch import nn

from cifar_resnet.checkpoint import load_model, read_checkpoint, save_weights
from cifar_resnet.config import Config, apply_overrides, from_dict, load_config
from cifar_resnet.data import build_train_val_loaders
from cifar_resnet.ema import ModelEMA
from cifar_resnet.engine import build_optimizer, build_scheduler, evaluate, train_one_epoch
from cifar_resnet.models import build_model, count_parameters
from cifar_resnet.rules import COMPETITION_MAX_PARAMS, check_teacher
from cifar_resnet.sam import SAM
from cifar_resnet.utils import get_device, needs_grad_scaler, seed_everything, setup_logging

log = logging.getLogger(__name__)

METRIC_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "ema_val_loss",
    "ema_val_acc",
    "epoch_time_s",
]


def _make_scaler(device: torch.device, amp: bool):
    if not needs_grad_scaler(device, amp):
        return None
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def _append_metrics(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def train(cfg: Config, resume: str | Path | None = None) -> dict:
    """Run a full training job and return its summary (also written to summary.json)."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    cfg.save(run_dir / "config.yaml")
    seed_everything(cfg.train.seed)
    device = get_device()
    log.info("Run %s on %s", cfg.run_name, device)

    train_loader, val_loader = build_train_val_loaders(cfg, device)
    channels_last = cfg.train.channels_last and device.type == "cuda"

    model = build_model(cfg.model).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = count_parameters(model)
    log.info("Model: %s", cfg.model)
    log.info("Trainable parameters: %s", f"{n_params:,}")
    if n_params > COMPETITION_MAX_PARAMS:
        log.warning(
            "This model is over the %s-parameter competition limit: it can only be used as a "
            "distillation teacher and will be refused by cifar-predict.",
            f"{COMPETITION_MAX_PARAMS:,}",
        )

    teacher = None
    if cfg.distill.teacher:
        teacher_ckpt = read_checkpoint(cfg.distill.teacher)
        check_teacher(teacher_ckpt, cfg.distill.teacher, cfg)
        teacher, _, _ = load_model(cfg.distill.teacher, device, enforce_budget=False)
        if channels_last:
            teacher = teacher.to(memory_format=torch.channels_last)
        log.info(
            "Distilling from %s (%s params, alpha %.2f, T %.1f)",
            cfg.distill.teacher,
            f"{count_parameters(teacher):,}",
            cfg.distill.alpha,
            cfg.distill.temperature,
        )

    steps_per_epoch = len(train_loader)
    if cfg.train.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.train.max_steps_per_epoch)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, steps_per_epoch, cfg.train.epochs)
    scaler = _make_scaler(device, cfg.train.amp)
    sam = None
    if cfg.optim.sam_rho > 0:
        if scaler is not None:
            raise ValueError(
                "SAM is not supported with fp16 loss scaling. This GPU lacks bf16; "
                "set train.amp=false to train in fp32."
            )
        sam = SAM(optimizer, rho=cfg.optim.sam_rho, adaptive=cfg.optim.sam_adaptive)
        log.info("Using %sSAM with rho=%s", "adaptive " if sam.adaptive else "", sam.rho)
    ema = ModelEMA(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing)

    start_epoch, best_acc, best_source = 0, -1.0, ""
    if resume is None and cfg.train.auto_resume and (run_dir / "last.pt").exists():
        resume = run_dir / "last.pt"
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema"):
            ema.load_state_dict(state["ema"])
        start_epoch, best_acc = state["epoch"] + 1, state.get("best_acc", -1.0)
        best_source = state.get("best_source", "")
        log.info("Resumed from %s at epoch %d", resume, start_epoch)

    metrics_path = run_dir / "metrics.csv"
    if start_epoch == 0 and metrics_path.exists():
        metrics_path.unlink()

    eval_kwargs = {"tta": False, "amp": cfg.train.amp, "channels_last": channels_last}
    t_start = time.time()
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            cfg.augment,
            amp=cfg.train.amp,
            scaler=scaler,
            ema=ema,
            teacher=teacher,
            distill=cfg.distill,
            channels_last=channels_last,
            max_steps=cfg.train.max_steps_per_epoch,
            sam=sam,
        )
        row = {
            "epoch": epoch + 1,
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            "train_loss": f"{stats.loss:.4f}",
            "train_acc": f"{stats.acc:.4f}",
        }

        is_last = epoch + 1 == cfg.train.epochs
        if val_loader is not None and ((epoch + 1) % cfg.train.eval_every == 0 or is_last):
            val = evaluate(model, val_loader, device, **eval_kwargs)
            row.update(val_loss=f"{val.loss:.4f}", val_acc=f"{val.acc:.4f}")
            candidates = [(val.acc, "model", model)]
            if ema is not None:
                ema_val = evaluate(ema.module, val_loader, device, **eval_kwargs)
                row.update(ema_val_loss=f"{ema_val.loss:.4f}", ema_val_acc=f"{ema_val.acc:.4f}")
                candidates.append((ema_val.acc, "ema", ema.module))
            acc, source, best_module = max(candidates, key=lambda c: c[0])
            if acc > best_acc:
                best_acc, best_source = acc, source
                save_weights(
                    run_dir / "best.pt",
                    best_module,
                    cfg,
                    epoch=epoch + 1,
                    val_acc=acc,
                    source=source,
                )

        row["epoch_time_s"] = f"{time.time() - t0:.1f}"
        _append_metrics(metrics_path, row)
        log.info(
            "epoch %3d/%d | lr %s | train loss %s acc %s | val acc %s | ema val acc %s | %ss",
            epoch + 1,
            cfg.train.epochs,
            row["lr"],
            row["train_loss"],
            row["train_acc"],
            row.get("val_acc", "-"),
            row.get("ema_val_acc", "-"),
            row["epoch_time_s"],
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "ema": ema.state_dict() if ema is not None else None,
                "epoch": epoch,
                "best_acc": best_acc,
                "best_source": best_source,
                "config": cfg.to_dict(),
            },
            run_dir / "last.pt",
        )

    # Without a validation split (final run on all 50k images), export the EMA weights.
    final_module = ema.module if ema is not None else model
    save_weights(run_dir / "final.pt", final_module, cfg, epoch=cfg.train.epochs)
    if val_loader is None:
        best_source = "ema" if ema is not None else "model"
        save_weights(run_dir / "best.pt", final_module, cfg, epoch=cfg.train.epochs)

    summary = {
        "run_name": cfg.run_name,
        "params": n_params,
        "epochs": cfg.train.epochs,
        "train_images": len(train_loader.dataset),
        "best_val_acc": best_acc if val_loader is not None else None,
        "best_source": best_source,
        "train_time_min": round((time.time() - t_start) / 60, 1),
        "device": str(device),
    }
    if val_loader is not None:
        best_model = load_model(run_dir / "best.pt", device)[0]
        tta = evaluate(best_model, val_loader, device, tta=True, amp=cfg.train.amp)
        summary["best_val_acc_tta"] = tta.acc
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary: %s", json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", help="YAML experiment config")
    parser.add_argument("--resume", help="resume from a run's last.pt (uses its saved config)")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.epochs=50")
    args = parser.parse_args(argv)

    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
        cfg = apply_overrides(from_dict(saved), args.overrides)
    else:
        cfg = load_config(args.config, args.overrides)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
