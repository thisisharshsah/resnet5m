from __future__ import annotations

import torch
from torch import nn


class LogitEnsemble(nn.Module):
    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members]).mean(0)
