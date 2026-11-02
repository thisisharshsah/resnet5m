import pytest
import torch

import cifar_resnet.data as data_module
from cifar_resnet.checkpoint import save_weights
from cifar_resnet.config import ModelConfig, from_dict, load_config
from cifar_resnet.models import build_model
from cifar_resnet.predict import predict
from cifar_resnet.rules import COMPETITION_MAX_PARAMS, RuleViolation, check_teacher
from cifar_resnet.train import train


def _smoke_cfg(fake_data_root, tmp_path, *extra):
    return load_config(
        "configs/smoke.yaml",
        [
            f"data.root={fake_data_root}",
            f"data.cache_dir={tmp_path / 'cache'}",
            f"output_dir={tmp_path / 'runs'}",
            *extra,
        ],
    )


def test_predict_refuses_models_over_5m_params(fake_data_root, tmp_path):
    cfg = _smoke_cfg(fake_data_root, tmp_path)
    cfg.model = ModelConfig(channels=[128, 256, 512], blocks=[2, 2, 2], max_params=50_000_000)
    model = build_model(cfg.model)
    assert sum(p.numel() for p in model.parameters()) > COMPETITION_MAX_PARAMS
    save_weights(tmp_path / "big.pt", model, cfg)
    with pytest.raises(RuleViolation, match="limited to"):
        predict(tmp_path / "big.pt", tmp_path / "sub.csv", fake_data_root)


def test_predict_refuses_weights_not_trained_by_this_code(fake_data_root, tmp_path):
    cfg = _smoke_cfg(fake_data_root, tmp_path)
    model = build_model(cfg.model)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, tmp_path / "foreign.pt")
    with pytest.raises(RuleViolation, match="from scratch"):
        predict(tmp_path / "foreign.pt", tmp_path / "sub.csv", fake_data_root)


def test_teacher_must_not_have_seen_the_validation_split(fake_data_root, tmp_path):
    student = _smoke_cfg(fake_data_root, tmp_path, "data.val_size=20")
    teacher_cfg = _smoke_cfg(fake_data_root, tmp_path, "data.val_size=20", "data.split_seed=7")
    ckpt = {"origin": "cifar_resnet.train", "config": teacher_cfg.to_dict()}
    with pytest.raises(ValueError, match="validation split"):
        check_teacher(ckpt, "teacher.pt", student)
    teacher_cfg = _smoke_cfg(fake_data_root, tmp_path, "data.val_size=0")
    ckpt = {"origin": "cifar_resnet.train", "config": teacher_cfg.to_dict()}
    with pytest.raises(ValueError, match="validation split"):
        check_teacher(ckpt, "teacher.pt", student)

    full_student = from_dict(
        {**student.to_dict(), "data": {**student.to_dict()["data"], "val_size": 0}}
    )
    check_teacher(ckpt, "teacher.pt", full_student)


def test_training_never_reads_the_test_split(fake_data_root, tmp_path, monkeypatch):
    seen = []
    original = data_module.load_split

    def recording_load_split(root, split, cache_dir=None):
        seen.append(split)
        return original(root, split, cache_dir)

    monkeypatch.setattr(data_module, "load_split", recording_load_split)
    train(_smoke_cfg(fake_data_root, tmp_path))
    assert seen and set(seen) == {"train"}


def test_sam_and_ensemble_distillation_train_end_to_end(fake_data_root, tmp_path):
    for name, seed in (("teacher", 0), ("teacher_s1", 1)):
        train(_smoke_cfg(fake_data_root, tmp_path, f"run_name={name}", f"train.seed={seed}"))
    teachers = [str(tmp_path / "runs" / n / "best.pt") for n in ("teacher", "teacher_s1")]
    student_cfg = _smoke_cfg(
        fake_data_root,
        tmp_path,
        "run_name=student",
        "optim.sam_rho=0.05",
        "augment.mix=cutmix",
        "augment.mix_prob=1.0",
        f"distill.teacher={teachers}",
    )
    summary = train(student_cfg)
    assert summary["params"] <= COMPETITION_MAX_PARAMS
    sub = predict(tmp_path / "runs" / "student" / "best.pt", tmp_path / "sub.csv")
    assert len(sub) == 12
