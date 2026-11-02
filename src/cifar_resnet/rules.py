from __future__ import annotations

from typing import Any

from torch import nn

from cifar_resnet.config import Config, from_dict
from cifar_resnet.models import count_parameters

COMPETITION_MAX_PARAMS = 5_000_000

CHECKPOINT_ORIGIN = "cifar_resnet.train"


class RuleViolation(RuntimeError):
    pass


def check_submission_model(model: nn.Module) -> int:
    n_params = count_parameters(model)
    if n_params > COMPETITION_MAX_PARAMS:
        raise RuleViolation(
            f"Model has {n_params:,} trainable parameters; submissions are limited to "
            f"{COMPETITION_MAX_PARAMS:,}. Over-budget models may only be used as teachers."
        )
    return n_params


def check_trained_from_scratch(ckpt: dict[str, Any], path: str) -> None:
    if ckpt.get("origin") != CHECKPOINT_ORIGIN:
        raise RuleViolation(
            f"{path} was not produced by cifar_resnet.train. Only models trained from scratch "
            "on the permitted data with this codebase can be used (no pretrained weights)."
        )


def check_teacher(ckpt: dict[str, Any], path: str, student: Config) -> Config:
    check_trained_from_scratch(ckpt, path)
    teacher = from_dict(ckpt["config"])
    if student.data.val_size > 0:
        same_split = (
            teacher.data.val_size == student.data.val_size
            and teacher.data.split_seed == student.data.split_seed
        )
        if not same_split:
            raise ValueError(
                f"Teacher {path} was trained with val_size={teacher.data.val_size}, "
                f"split_seed={teacher.data.split_seed}, so it saw images in this run's "
                f"validation split (val_size={student.data.val_size}, "
                f"split_seed={student.data.split_seed}). The student's validation accuracy "
                "would be optimistic. Train the teacher on the same split."
            )
    return teacher
