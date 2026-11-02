from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class SAM:
    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05, adaptive: bool = False):
        if rho <= 0:
            raise ValueError("rho must be positive")
        self.optimizer = optimizer
        self.rho = rho
        self.adaptive = adaptive
        self._perturbations: dict[torch.Tensor, torch.Tensor] = {}

    def _params(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    yield p

    @torch.no_grad()
    def first_step(self) -> None:
        norms = [((p.abs() if self.adaptive else 1.0) * p.grad).norm(2) for p in self._params()]
        grad_norm = torch.stack(norms).norm(2)
        scale = self.rho / (grad_norm + 1e-12)
        for p in self._params():
            e_w = (p.pow(2) if self.adaptive else 1.0) * p.grad * scale
            p.add_(e_w)
            self._perturbations[p] = e_w

    @torch.no_grad()
    def second_step(self) -> None:
        for p, e_w in self._perturbations.items():
            p.sub_(e_w)
        self._perturbations.clear()
        self.optimizer.step()


@contextmanager
def frozen_bn_stats(model: nn.Module):
    saved = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            saved.append((m, m.momentum))
            m.momentum = 0.0
    try:
        yield
    finally:
        for m, momentum in saved:
            m.momentum = momentum
