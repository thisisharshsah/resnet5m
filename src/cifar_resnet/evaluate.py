from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cifar_resnet.checkpoint import load_model
from cifar_resnet.data import CLASSES, build_val_split
from cifar_resnet.engine import TTA_MODES, predict_logits
from cifar_resnet.models import count_parameters
from cifar_resnet.utils import get_device, setup_logging

log = logging.getLogger(__name__)


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path | None = None,
    modes: tuple[str, ...] = TTA_MODES,
    output: str | Path | None = None,
) -> dict:
    device = get_device()
    model, cfg, ckpt = load_model(checkpoint, device, enforce_budget=False)
    if data_root is not None:
        cfg.data.root = str(data_root)
    if cfg.data.val_size <= 0:
        raise ValueError(f"{checkpoint} was trained on all images; it has no validation split")
    dataset = build_val_split(cfg)
    loader = DataLoader(dataset, batch_size=512, num_workers=min(4, cfg.data.num_workers))

    result = {
        "checkpoint": str(checkpoint),
        "run_name": cfg.run_name,
        "params": count_parameters(model),
        "epoch": ckpt.get("epoch"),
        "val_size": len(dataset),
        "accuracy": {},
    }
    for mode in modes:
        logits, labels = predict_logits(model, loader, device, tta=mode, amp=cfg.train.amp)
        pred = logits.argmax(1)
        result["accuracy"][mode] = (pred == labels).float().mean().item()
        log.info("%s | tta=%-5s | val acc %.4f", cfg.run_name, mode, result["accuracy"][mode])

    labels_np, pred_np = labels.numpy(), pred.numpy()
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(confusion, (labels_np, pred_np), 1)
    result["confusion"] = confusion.tolist()
    result["confusion_tta"] = modes[-1]
    result["per_class_accuracy"] = {
        name: float(confusion[i, i] / max(1, confusion[i].sum())) for i, name in enumerate(CLASSES)
    }
    result["classes"] = list(CLASSES)

    output = Path(output) if output else Path(checkpoint).parent / "val_eval.json"
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", output)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, nargs="+")
    parser.add_argument("--data-root")
    parser.add_argument("--tta", nargs="+", choices=TTA_MODES, default=list(TTA_MODES))
    args = parser.parse_args(argv)
    setup_logging()
    for ckpt in args.checkpoint:
        evaluate_checkpoint(ckpt, args.data_root, tuple(args.tta))


if __name__ == "__main__":
    main()
