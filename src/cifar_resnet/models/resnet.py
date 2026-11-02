from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: float) -> None:
        super().__init__()
        hidden = max(8, int(channels * ratio))
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        skip_kernel_size: int = 1,
        se_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels, se_ratio) if se_ratio > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    skip_kernel_size,
                    stride,
                    skip_kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        blocks: Sequence[int],
        kernel_size: int = 3,
        skip_kernel_size: int = 1,
        stem_channels: int | None = None,
        pool_size: int | None = None,
        dropout: float = 0.0,
        se_ratio: float = 0.0,
        zero_init_residual: bool = True,
        num_classes: int = 10,
        input_size: int = 32,
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks) or not channels:
            raise ValueError("channels and blocks must be non-empty and the same length")
        if kernel_size % 2 == 0 or skip_kernel_size % 2 == 0:
            raise ValueError("kernel sizes must be odd so padding keeps spatial size")

        stem_channels = stem_channels or channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = stem_channels
        for i, (out_ch, n_blocks) in enumerate(zip(channels, blocks)):
            layers = []
            for j in range(n_blocks):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(
                    BasicBlock(in_ch, out_ch, stride, kernel_size, skip_kernel_size, se_ratio)
                )
                in_ch = out_ch
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1) if pool_size is None else nn.AvgPool2d(pool_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        with torch.no_grad():
            feat = self.pool(self.stages(self.stem(torch.zeros(1, 3, input_size, input_size))))
        self.fc = nn.Linear(feat.numel(), num_classes)

        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.stages(self.stem(x))), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))
