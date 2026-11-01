"""Exponential moving average of model weights (Polyak averaging)."""

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Warm up the decay so early, noisy weights are forgotten quickly.
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(d).add_(value, alpha=1 - d)
            else:
                ema_value.copy_(value)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = state["updates"]
