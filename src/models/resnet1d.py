"""Hannun-style 1D ResNet (SOTA #1).

Loose reimplementation of the deep 1D residual network introduced by Hannun
et al. (*Nature Medicine*, 2019) for arrhythmia classification from
single-lead ECG. The original network is 33 layers deep with kernel width
16 and a constant filter count after the stem; we shrink the depth
substantially (8 residual blocks instead of 16) and use a channel-doubling
ladder that *plateaus at 128*. The natural ladder would continue to 256
filters in the last three blocks, but that pushes the parameter count to
~6.6 M — over the 5 M cap CLAUDE.md sets for plausible wearable
deployment. Capping at 128 keeps the depth and downsampling schedule
identical and lands at ~3.1 M parameters. Conceptually we are closer to
the original constant-width Hannun design — we just plateau earlier than
the canonical doubling ladder would.

The block design is the canonical "two convs, skip connection, ReLU
after the sum" residual block. Every other block downsamples temporally
with stride 2 on the first conv and a strided 1x1 shortcut on the skip
path. The result is a roughly logarithmic spatial schedule:
15000 -> 7500 -> 469 over the stem and four downsampling blocks.

Input:  (B, 2, 15000)
Output: (B,) — one pre-sigmoid logit per window.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    """Two-conv residual block with optional strided 1x1 shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # The shortcut needs a 1x1 conv whenever the main path changes the
        # tensor shape — either via channel growth or temporal downsampling.
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, bias=False,
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# (in_ch, out_ch, stride) per block, in order. The doubling ladder is
# truncated at 128 channels — blocks 6 and 8 still downsample temporally,
# but the channel width stops growing past 128. See the module docstring
# for why (parameter budget).
_BLOCK_PLAN: tuple[tuple[int, int, int], ...] = (
    (32, 32, 1),
    (32, 64, 2),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 128, 2),
    (128, 128, 1),
    (128, 128, 2),
)


class ResNet1D(nn.Module):
    """Conv stem -> 8 residual blocks -> global average pool -> linear head."""

    def __init__(
        self,
        in_channels: int = 2,
        stem_channels: int = 32,
        kernel_size: int = 15,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Stem: (B, 2, 15000) -> (B, 32, 7500). Stride 2 halves the temporal
        # dimension up front so the deeper blocks don't have to.
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, stem_channels,
                kernel_size=kernel_size, stride=2,
                padding=kernel_size // 2, bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock1D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            )
            for (in_ch, out_ch, stride) in _BLOCK_PLAN
        ])

        final_channels = _BLOCK_PLAN[-1][1]
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(final_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)        # (B, 128)
        logit = self.head(x).squeeze(-1)           # (B,)
        return logit
