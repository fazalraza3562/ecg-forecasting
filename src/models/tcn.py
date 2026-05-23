"""Temporal Convolutional Network (Bai et al., 2018) for ECG forecasting (SOTA #3).

The TCN argues that a sufficiently deep stack of dilated convolutions can
match or beat recurrent models on long-range sequence tasks while being
fully parallelisable along the time axis. We follow the Bai 2018 recipe
with two deviations made explicit in the code:

* **Symmetric (non-causal) padding.** The original TCN is causal — every
  output at time ``t`` is computed from inputs at times ``<= t`` by
  left-padding each conv with ``(k-1)*d`` zeros. That matters for
  autoregressive generation. We are *not* autoregressive; the forecasting
  question is "given the whole 60 s past window, does VT start in the
  next 30 s?" so there is no benefit to enforcing causality and a cost
  in receptive-field utilisation at the window boundaries. We use
  symmetric padding ``(k-1)*d // 2`` on both sides instead, which is the
  standard non-causal TCN variant.
* **Constant channel width (64).** The paper grows channel count with
  depth. We hold it at 64 throughout to keep the parameter count under
  the 5 M cap CLAUDE.md sets; growing channels is mostly a representation-
  capacity lever and 64 is already wider than the per-block bottleneck we
  found necessary on this task.

Other choices follow the paper: kernel=7, two stacked dilated convs per
block, dilations doubling per block (1, 2, 4, 8, 16, 32) for an exponential
receptive field, ReLU activation (not GELU), weight normalisation on the
dilated convs (via the modern ``torch.nn.utils.parametrizations.weight_norm``
API; the legacy ``torch.nn.utils.weight_norm`` was deprecated in 2.1),
spatial dropout 0.2, and a residual connection per block.

Receptive field at the final layer is 2 * (k-1) * sum(dilations) = 2 * 6 *
63 = 756 post-stem samples, or roughly 3 s of ECG context, which comfortably
covers several beats at the 250 Hz sample rate / 4x stem stride.

Input:  (B, 2, 15000)
Output: (B,) — one pre-sigmoid logit per window.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


# Per-block dilation factors. Doubling gives an exponentially growing
# receptive field with linear depth.
_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


class TemporalBlock(nn.Module):
    """Two dilated weight-normed convs with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Symmetric padding so the temporal length is preserved by each conv.
        # See module docstring for why we use symmetric (non-causal) padding.
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, padding=padding,
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, padding=padding,
        ))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # The spec keeps channels constant so this is Identity in practice;
        # the 1x1 shortcut is kept for safety if the channel plan ever
        # changes downstream.
        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv1d(
                in_channels, out_channels, kernel_size=1,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + identity


class TCN1D(nn.Module):
    """Conv stem -> 6 dilated TemporalBlocks -> global average pool -> linear head."""

    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.2,
        dilations: tuple[int, ...] = _DILATIONS,
    ) -> None:
        super().__init__()
        # (B, 2, 15000) -> (B, 64, 3750). One stride-4 cut up front so the
        # dilation=32 layer at the end has a manageable receptive-field stride.
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, channels,
                kernel_size=kernel_size, stride=4,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            TemporalBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            for d in dilations
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)        # (B, 64)
        logit = self.head(x).squeeze(-1)           # (B,)
        return logit
