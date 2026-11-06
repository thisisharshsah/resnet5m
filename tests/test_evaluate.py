import json

from cifar_resnet.config import load_config
from cifar_resnet.evaluate import evaluate_checkpoint
from cifar_resnet.train import train


def test_evaluate_writes_confusion_and_accuracies(fake_data_root, tmp_path):
    cfg = load_config(
        "configs/smoke.yaml",
        [
            f"data.root={fake_data_root}",
            f"data.cache_dir={tmp_path / 'cache'}",
            f"output_dir={tmp_path / 'runs'}",
            "train.epochs=1",
        ],
    )
    train(cfg)
    result = evaluate_checkpoint(tmp_path / "runs" / "smoke" / "best.pt")
    saved = json.loads((tmp_path / "runs" / "smoke" / "val_eval.json").read_text())
    assert set(result["accuracy"]) == {"none", "flip", "shift"}
    assert all(0.0 <= v <= 1.0 for v in result["accuracy"].values())
    assert len(saved["confusion"]) == 10 and sum(map(sum, saved["confusion"])) == result["val_size"]
    assert set(saved["per_class_accuracy"]) == set(saved["classes"])
