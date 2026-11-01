"""Competition rules, enforced in code.

Each rule that code can check is checked here and raises ``RuleViolation``, so a mistake
fails loudly instead of producing an invalid submission. See "Rules compliance" in the README
for how every rule in the brief is handled.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from cifar_resnet.config import Config, from_dict
from cifar_resnet.models import count_parameters

COMPETITION_MAX_PARAMS = 5_000_000
"""Hard limit on trainable parameters for any submitted model (not configurable)."""

CHECKPOINT_ORIGIN = "cifar_resnet.train"
"""Stamped into every checkpoint this code trains, so only our own weights can be reused."""


class RuleViolation(RuntimeError):
    """Raised when an action would break the competition rules."""


def check_submission_model(model: nn.Module) -> int:
    """A submitted model is a single network with at most 5M trainable parameters."""
    n_params = count_parameters(model)
    if n_params > COMPETITION_MAX_PARAMS:
        raise RuleViolation(
            f"Model has {n_params:,} trainable parameters; submissions are limited to "
            f"{COMPETITION_MAX_PARAMS:,}. Over-budget models may only be used as teachers."
        )
    return n_params


def check_trained_from_scratch(ckpt: dict[str, Any], path: str) -> None:
    """Reject weights not produced by our own training code (e.g. downloaded or pretrained)."""
    if ckpt.get("origin") != CHECKPOINT_ORIGIN:
        raise RuleViolation(
            f"{path} was not produced by cifar_resnet.train. Only models trained from scratch "
            "on the permitted data with this codebase can be used (no pretrained weights)."
        )


def check_teacher(ckpt: dict[str, Any], path: str, student: Config) -> Config:
    """A distillation teacher must be our own from-scratch model, and, when the student is
    validated on a held-out split, the teacher must not have trained on that split."""
    check_trained_from_scratch(ckpt, path)
    teacher = from_dict(ckpt["config"])
    if student.data.val_size > 0:
        same_split = (
            teacher.data.val_size == student.data.val_size
            and teacher.train.seed == student.train.seed
        )
        if not same_split:
            raise ValueError(
                f"Teacher {path} was trained with val_size={teacher.data.val_size}, "
                f"seed={teacher.train.seed}, so it saw images in this run's validation split "
                f"(val_size={student.data.val_size}, seed={student.train.seed}). The student's "
                "validation accuracy would be optimistic. Train the teacher on the same split."
            )
    return teacher
