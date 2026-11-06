from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from cifar_resnet.checkpoint import load_model, read_checkpoint
from cifar_resnet.data import build_test_loader, find_data_root
from cifar_resnet.engine import TTA_MODES, predict_logits
from cifar_resnet.rules import check_submission_model, check_trained_from_scratch
from cifar_resnet.utils import get_device, setup_logging

log = logging.getLogger(__name__)


def write_submission(ids, labels, path: str | Path, data_root: str | Path | None = None) -> Path:
    sub = pd.DataFrame({"id": ids, "label": labels}).sort_values("id").reset_index(drop=True)
    if not sub["label"].between(0, 9).all():
        raise ValueError("labels must be integers in [0, 9]")
    if data_root is not None:
        sample = find_data_root(data_root) / "sample_submission.csv"
        if sample.exists():
            expected = pd.read_csv(sample)["id"].sort_values().to_numpy()
            if not (len(expected) == len(sub) and (expected == sub["id"].to_numpy()).all()):
                raise ValueError("submission ids do not match sample_submission.csv")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def predict(
    checkpoint: str | Path,
    output: str | Path = "submission.csv",
    data_root: str | Path | None = None,
    tta: bool | str = "flip",
    save_logits: bool = True,
) -> pd.DataFrame:
    device = get_device()
    check_trained_from_scratch(read_checkpoint(checkpoint), str(checkpoint))
    model, cfg, ckpt = load_model(checkpoint, device, enforce_budget=False)
    n_params = check_submission_model(model)
    root = data_root or cfg.data.root
    log.info(
        "Loaded %s (%s params, epoch %s), tta=%s",
        checkpoint,
        f"{n_params:,}",
        ckpt.get("epoch"),
        tta,
    )

    loader, ids = build_test_loader(root, cfg.data.cache_dir, device=device)
    logits, _ = predict_logits(model, loader, device, tta=tta, amp=cfg.train.amp)
    labels = logits.argmax(1).numpy()
    path = write_submission(ids, labels, output, root)
    if save_logits:
        torch.save({"ids": ids, "logits": logits}, Path(path).with_suffix(".logits.pt"))
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    log.info("Wrote %d predictions to %s | class counts %s", len(labels), path, counts)
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--data-root", help="defaults to the data.root saved in the checkpoint")
    parser.add_argument("--tta", choices=TTA_MODES, default="flip")
    args = parser.parse_args(argv)
    setup_logging()
    predict(args.checkpoint, args.output, args.data_root, tta=args.tta)


if __name__ == "__main__":
    main()
