"""Vanilla Transformer encoder, scaled down for ECG forecasting.

We follow the standard Vaswani encoder recipe (multi-head self-attention +
position-wise feed-forward, with residual connections and layer norm) with
two ECG-specific changes:

* **Conv stem.** A bare token-per-sample encoder would mean self-attention
  over 15000 tokens, which is far past what we can afford. A single
  ``Conv1d(2, 64, kernel=15, stride=8)`` collapses the input into a
  sequence of 1874 tokens with d_model=64; one token now corresponds to
  ~32 ms of ECG, which is the right granularity for beat-level reasoning.
* **Pre-norm (`norm_first=True`).** The original 2017 paper places layer
  norm *after* the residual addition (post-norm). Small post-norm
  transformers are notoriously unstable to train without warmup and a
  careful learning-rate schedule. The modern recommendation, and the
  default for nearly every recent implementation, is pre-norm — apply
  layer norm to the inputs of the sub-layer rather than the outputs.

Positional encoding is the classical sinusoidal formulation from the
original paper. We materialise it once at construction (sized to a fixed
``max_len`` chosen well above the post-stem sequence length) and register
it as a buffer so it travels with ``.to(device)`` without being treated
as a trainable parameter.

Input:  (B, 2, 15000)
Output: (B,) — one pre-sigmoid logit per window.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return a ``(max_len, d_model)`` sinusoidal PE matrix."""
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even for sinusoidal PE; got {d_model}")
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerEncoderModel(nn.Module):
    """Conv stem -> sinusoidal PE -> 3 pre-norm encoder layers -> mean pool -> linear head."""

    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        # Conv stem: (B, 2, 15000) -> (B, 64, 1874). Stride 8 with kernel 15
        # gives a per-token receptive field of ~60 ms — wide enough to cover a
        # full QRS complex but narrow enough that the encoder still sees beat
        # ordering rather than averaged shape.
        self.stem = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=15,
            stride=8,
        )

        # PE is fixed and non-trainable. Buffer means it moves with .to(device)
        # but doesn't appear in optimizer parameter groups.
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_positional_encoding(max_len, d_model),
            persistent=False,
        )
        self._max_len = max_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )
        # A final LayerNorm after the stack is conventional with pre-norm.
        # enable_nested_tensor is explicitly off: it's incompatible with
        # norm_first=True, and leaving the default on prints a noisy warning
        # on every construction.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, d_model, T') -> (B, T', d_model) for batch_first.
        tokens = self.stem(x).transpose(1, 2)
        seq_len = tokens.size(1)
        if seq_len > self._max_len:
            raise RuntimeError(
                f"sequence length {seq_len} exceeds positional encoding "
                f"capacity {self._max_len}; raise max_len"
            )
        # Slice the buffer to the actual sequence length and add. Broadcasting
        # over the batch dim is automatic: (T', d_model) -> (B, T', d_model).
        # The .to(dtype) keeps AMP/half-precision training honest.
        tokens = tokens + self.positional_encoding[:seq_len].to(tokens.dtype)

        encoded = self.encoder(tokens)             # (B, T', d_model)
        pooled = encoded.mean(dim=1)               # (B, d_model)
        logit = self.head(pooled).squeeze(-1)      # (B,)
        return logit
