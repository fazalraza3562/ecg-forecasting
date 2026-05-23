"""InceptionTime adapted to 1D ECG (SOTA #2).

Follows Ismail Fawaz et al. (*InceptionTime: Finding AlexNet for Time
Series Classification*, 2020) with the conventional adjustments for the
ECG forecasting setting.

The Inception module is the architectural idea: rather than commit to a
single receptive-field size, run several parallel convolutions of
different kernel widths over the same input and let the network decide
which one to lean on. We use the canonical recipe for time-series
Inception — kernel sizes 9, 19, 39 (logarithmically spaced) on the three
"learned" branches, plus a max-pool branch that preserves un-bottlenecked
context. A 1x1 bottleneck before the wide kernels keeps the parameter
count flat as channel width grows; the pool branch deliberately skips
the bottleneck so the un-filtered input still has a path through the
module.

We use three Inception modules in series (the paper's residual variant
uses six, but three is enough at our spatial resolution and keeps us
well under the 5 M parameter cap). Channel width is 128 throughout
(4 branches x 32 channels). The conv stem cuts the spatial dimension
from 15000 to 3750 once up front so the kernel-39 branch is reading a
~150 ms window per output sample rather than ~150 / 4 ms.

Input:  (B, 2, 15000)
Output: (B,) — one pre-sigmoid logit per window.
"""
from __future__ import annotations

import torch
from torch import nn


# Kernel sizes for the three "learned" branches. Each must have padding
# k // 2 so the output length matches the input.
_BRANCH_KERNELS: tuple[int, ...] = (9, 19, 39)
_BRANCH_CHANNELS: int = 32
_BOTTLENECK_CHANNELS: int = 32


class InceptionBlock1D(nn.Module):
    """Four-branch inception module: 3 wide convs over a bottleneck + 1 pool branch."""

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = _BOTTLENECK_CHANNELS,
        branch_channels: int = _BRANCH_CHANNELS,
        kernel_sizes: tuple[int, ...] = _BRANCH_KERNELS,
    ) -> None:
        super().__init__()
        # 1x1 channel compression before the expensive wide kernels. No bias
        # because BatchNorm follows the concatenation.
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels,
            kernel_size=1, bias=False,
        )

        # Three parallel branches at increasing receptive-field width. Padding
        # is chosen so every branch output keeps the input length unchanged,
        # which is what makes the channel-concat at the end legal.
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                bottleneck_channels, branch_channels,
                kernel_size=k, padding=k // 2, bias=False,
            )
            for k in kernel_sizes
        ])

        # The pool branch sees the un-bottlenecked input — that's the whole
        # point of having it. MaxPool(k=3, s=1, p=1) preserves length; the
        # following 1x1 projects whatever input channel count down to
        # branch_channels so the concat dimensions line up.
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )

        out_channels = (len(kernel_sizes) + 1) * branch_channels  # 128
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottlenecked = self.bottleneck(x)
        branches = [conv(bottlenecked) for conv in self.conv_branches]
        branches.append(self.pool_branch(x))
        out = torch.cat(branches, dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class InceptionTime1D(nn.Module):
    """Conv stem -> 3 inception blocks -> global average pool -> linear head."""

    def __init__(
        self,
        in_channels: int = 2,
        stem_channels: int = 32,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        # (B, 2, 15000) -> (B, 32, 3750). One stride-4 cut up front so the
        # kernel-39 branch is reading ECG at a sensible time scale.
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, stem_channels,
                kernel_size=15, stride=4, padding=7, bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Block 1 reads the stem's stem_channels; subsequent blocks read the
        # 4 * branch_channels = 128 channels produced by the previous block.
        block_out_channels = (len(_BRANCH_KERNELS) + 1) * _BRANCH_CHANNELS
        blocks: list[nn.Module] = []
        prev_channels = stem_channels
        for _ in range(num_blocks):
            blocks.append(InceptionBlock1D(in_channels=prev_channels))
            prev_channels = block_out_channels
        self.blocks = nn.ModuleList(blocks)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(block_out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)        # (B, 128)
        logit = self.head(x).squeeze(-1)           # (B,)
        return logit
