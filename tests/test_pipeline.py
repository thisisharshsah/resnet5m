import json

import pandas as pd

from cifar_resnet.config import load_config
from cifar_resnet.predict import predict
from cifar_resnet.train import train


def test_train_then_predict(fake_data_root, tmp_path):
    cfg = load_config(
        "configs/smoke.yaml",
        [
            f"data.root={fake_data_root}",
            f"data.cache_dir={tmp_path / 'cache'}",
            f"output_dir={tmp_path / 'runs'}",
            "augment.mix=cutmix",
            "augment.mix_prob=1.0",
        ],
    )
    summary = train(cfg)
    run_dir = tmp_path / "runs" / "smoke"
    for name in ("best.pt", "last.pt", "final.pt", "metrics.csv", "config.yaml", "summary.json"):
        assert (run_dir / name).exists(), name
    assert json.loads((run_dir / "summary.json").read_text())["params"] == summary["params"]

    sub = predict(run_dir / "best.pt", tmp_path / "submission.csv")
    assert list(sub.columns) == ["id", "label"]
    assert sub["id"].tolist() == list(range(12))
    assert sub["label"].between(0, 9).all()


def test_resume_continues_training(fake_data_root, tmp_path):
    overrides = [
        f"data.root={fake_data_root}",
        f"data.cache_dir={tmp_path / 'cache'}",
        f"output_dir={tmp_path / 'runs'}",
        "train.epochs=2",
    ]
    cfg = load_config("configs/smoke.yaml", overrides)
    train(cfg)
    run_dir = tmp_path / "runs" / "smoke"
    cfg.train.epochs = 3
    train(cfg, resume=run_dir / "last.pt")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics["epoch"].tolist() == [1, 2, 3]
