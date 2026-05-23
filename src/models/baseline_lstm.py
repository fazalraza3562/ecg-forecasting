"""Stacked BiLSTM baseline.

The plainest sequence model we use: a single Conv1d stem followed by a
two-layer bidirectional LSTM and a mean-pool / linear head. There is no
original "paper" to defer to — this is the canonical recurrent baseline
that every ECG forecasting study includes, and it sets the floor that the
proposed CNN-LSTM-Attention model and the SOTA comparators have to beat.

A bare LSTM over the raw 15000-sample window is computationally pointless:
the recurrence would have to backprop through 15k time steps and the
information density per sample is tiny at 250 Hz. The Conv1d stem
(kernel=15, stride=8) gives the LSTM a sequence of length ~1874 with a
small learned receptive field per token, which is roughly the QRS width
that any downstream temporal model wants to see anyway.

Input:  (B, 2, 15000) — two leads, 60 s at 250 Hz.
Output: (B,)          — one pre-sigmoid logit per window.
"""
from __future__ import annotations

import torch
from torch import nn


class BaselineLSTM(nn.Module):
    """Conv1d stem -> 2-layer BiLSTM -> mean pool -> linear head."""

    def __init__(
        self,
        in_channels: int = 2,
        stem_channels: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        lstm_dropout: float = 0.2,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=15,
            stride=8,
        )
        self.lstm = nn.LSTM(
            input_size=stem_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(2 * hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> stem -> (B, C', T')
        feats = self.stem(x)
        # LSTM wants (B, T', C').
        feats = feats.transpose(1, 2)
        seq, _ = self.lstm(feats)
        # Mean-pool over time, then a small linear head produces the logit.
        pooled = seq.mean(dim=1)
        pooled = self.head_dropout(pooled)
        logit = self.head(pooled).squeeze(-1)
        return logit
