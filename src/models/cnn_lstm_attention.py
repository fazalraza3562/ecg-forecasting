"""CNN-LSTM-Attention — the proposed model, with a ``use_attention`` toggle.

The model the report's contribution is built on. Setting ``use_attention=False``
yields a mean-pooled ablation that keeps the CNN-LSTM stack identical but
replaces the attention head with uniform temporal averaging, so the report
can isolate what the attention layer is actually contributing.

The motivation is that the
three families of feature ECG forecasting needs — local morphology (QRS
shape, ST changes), short-range temporal context (the last few beats), and
selective focus on the part of the window most predictive of an imminent
arrhythmia — map naturally onto three sub-networks:

* A small 1D CNN stack as a learnable feature extractor. Three Conv -> BN
  -> ReLU -> MaxPool(4) blocks compress 15000 samples down to 234 feature
  vectors of width 128. The first conv uses kernel=15 (~60 ms at 250 Hz)
  to span a full QRS complex; the next two use kernel=7 to combine those
  beat-level features into rhythm-level features.
* A single-layer bidirectional LSTM (hidden=64) over the 234-step CNN
  output. This is the "short-range temporal context" piece; we need it
  because the morphology of a single beat does not predict imminent VT,
  but the *progression* of beats (couplets, R-on-T, sympathetic surges)
  does.
* Additive (Bahdanau-style) attention pooling. A learnable query vector
  decides which of the 234 BiLSTM time steps to weight when forming the
  fixed-size context vector. The attention weights are stored on
  ``self.last_attn_weights`` after each forward pass so the explainability
  notebooks can overlay them on the raw ECG.

Inputs and outputs follow the project-wide contract: ``forward`` takes a
``(B, 2, 15000)`` tensor and returns a ``(B,)`` pre-sigmoid logit tensor.
"""
from __future__ import annotations

import torch
from torch import nn


class CNNLSTMAttention(nn.Module):
    """3 Conv blocks -> BiLSTM -> additive attention -> linear head."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_size: int = 64,
        attn_hidden: int = 64,
        head_dropout: float = 0.3,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        # Stride=1 on every conv; the MaxPool1d(4) layers do all the downsampling.
        # Padding keeps the length identical until the pool, which makes the 4x
        # ratio per block exact and the final T' (234) easy to reason about.
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(4)
        self.relu = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        h_dim = 2 * hidden_size  # 128: concat of forward and backward hidden states.

        # Bahdanau-style additive attention: score_t = v^T tanh(W h_t + W q),
        # where q is a learnable query in the same space as h_t and W is the
        # shared projection into the lower-dimensional attention space. This is
        # the standard formulation but with W reused across h and q, which both
        # halves the parameter count and makes "q lives in h-space" precise.
        #
        # We only allocate these parameters when use_attention is on. The
        # ablation variant pools by mean and therefore has no attention
        # parameters to learn — checking the param count is the simplest
        # way to confirm the ablation is structurally honest.
        if use_attention:
            self.attn_W = nn.Linear(h_dim, attn_hidden, bias=False)
            self.attn_query = nn.Parameter(torch.zeros(h_dim))
            self.attn_v = nn.Linear(attn_hidden, 1, bias=False)

        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(h_dim, 1)

        # Populated by forward(). Detached so callers can read it without
        # accidentally extending the autograd graph.
        self.last_attn_weights: torch.Tensor | None = None

    def _conv_block(
        self,
        x: torch.Tensor,
        conv: nn.Conv1d,
        bn: nn.BatchNorm1d,
    ) -> torch.Tensor:
        return self.pool(self.relu(bn(conv(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 2, 15000) -> (B, 32, 3750) -> (B, 64, 937) -> (B, 128, 234)
        x = self._conv_block(x, self.conv1, self.bn1)
        x = self._conv_block(x, self.conv2, self.bn2)
        x = self._conv_block(x, self.conv3, self.bn3)

        # (B, C, T') -> (B, T', C) for the LSTM.
        x = x.transpose(1, 2)
        seq, _ = self.lstm(x)  # (B, T', 128)

        if self.use_attention:
            h_proj = self.attn_W(seq)                          # (B, T', attn_hidden)
            q_proj = self.attn_W(self.attn_query)              # (attn_hidden,)
            scores = self.attn_v(torch.tanh(h_proj + q_proj))  # (B, T', 1)
            weights = scores.softmax(dim=1)                    # (B, T', 1)
            context = (weights * seq).sum(dim=1)               # (B, 128)
            # Cache for the explainability notebooks; shape (B, T').
            self.last_attn_weights = weights.squeeze(-1).detach()
        else:
            # Uniform pooling baseline: every time step contributes equally
            # to the context vector, so the model has to rely on the CNN
            # and LSTM alone for selectivity.
            context = seq.mean(dim=1)                          # (B, 128)

        pooled = self.head_dropout(context)
        logit = self.head(pooled).squeeze(-1)
        return logit
