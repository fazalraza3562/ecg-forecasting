"""Smoke-test every model: a random batch goes in, a (B,) logit tensor comes out.

These are intentionally minimal. The goal is to catch shape bugs and obvious
numerical blow-ups at import-time, not to assess model quality. One test per
architecture, each with a tiny random batch.
"""
from __future__ import annotations

import torch


# Shape of the real input window: 2 leads, 60 s at 250 Hz = 15000 samples.
BATCH_SIZE = 2
N_CHANNELS = 2
N_SAMPLES = 15000
PARAM_BUDGET = 5_000_000  # CLAUDE.md cap; lightweight enough for wearables.


def _make_batch() -> torch.Tensor:
    # Fixed seed so a flaky failure can be reproduced from the test name alone.
    g = torch.Generator().manual_seed(0)
    return torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES, generator=g)


def _assert_logit_tensor(out: torch.Tensor) -> None:
    assert out.shape == (BATCH_SIZE,), f"expected ({BATCH_SIZE},), got {tuple(out.shape)}"
    assert out.dtype.is_floating_point, f"expected float dtype, got {out.dtype}"
    assert torch.isfinite(out).all(), "output contains NaN or Inf"


def _assert_under_param_budget(model: torch.nn.Module) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < PARAM_BUDGET, f"model has {n_params:,} params; budget is {PARAM_BUDGET:,}"


def test_baseline_lstm() -> None:
    from src.models.baseline_lstm import BaselineLSTM

    model = BaselineLSTM().eval()
    _assert_under_param_budget(model)
    with torch.no_grad():
        out = model(_make_batch())
    _assert_logit_tensor(out)


def test_cnn_lstm_attention() -> None:
    from src.models.cnn_lstm_attention import CNNLSTMAttention

    model = CNNLSTMAttention().eval()
    _assert_under_param_budget(model)
    with torch.no_grad():
        out = model(_make_batch())
    _assert_logit_tensor(out)

    # The attention pooling is the model's distinguishing piece; check both
    # that the cached weights are exposed for the explainability notebooks
    # and that the softmax normalisation actually fires.
    attn = model.last_attn_weights
    assert attn is not None, "expected last_attn_weights to be populated by forward()"
    assert attn.ndim == 2, f"expected 2D (B, T') attn weights, got shape {tuple(attn.shape)}"
    assert attn.shape[0] == BATCH_SIZE, f"batch dim mismatch: {attn.shape[0]} vs {BATCH_SIZE}"
    row_sums = attn.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"attention weights should sum to 1 per row, got {row_sums.tolist()}"
    )


def test_cnn_lstm_noattention() -> None:
    from src.models.cnn_lstm_attention import CNNLSTMAttention

    model = CNNLSTMAttention(use_attention=False).eval()
    _assert_under_param_budget(model)
    with torch.no_grad():
        out = model(_make_batch())
    _assert_logit_tensor(out)

    # Mean-pool variant: no attention weights cached.
    assert model.last_attn_weights is None, (
        f"expected last_attn_weights=None when use_attention=False, "
        f"got {model.last_attn_weights}"
    )

    # The ablation must actually drop parameters relative to the full model
    # — otherwise we'd be lying about what's being ablated.
    full_params = sum(p.numel() for p in CNNLSTMAttention().parameters())
    abl_params = sum(p.numel() for p in model.parameters())
    assert abl_params < full_params, (
        f"ablation must drop attention params; got abl={abl_params:,} >= full={full_params:,}"
    )


def test_transformer() -> None:
    from src.models.transformer import TransformerEncoderModel

    model = TransformerEncoderModel().eval()
    _assert_under_param_budget(model)
    with torch.no_grad():
        out = model(_make_batch())
    _assert_logit_tensor(out)


def test_resnet1d() -> None:
    from src.models.resnet1d import ResNet1D

    model = ResNet1D().eval()
    _assert_under_param_budget(model)
    with torch.no_grad():
        out = model(_make_batch())
    _assert_logit_tensor(out)


def test_inception1d() -> None:
    from src.models.inception1d import InceptionTime1D

    model = InceptionTime1D().eval()
    _assert_under_param_budget(model)
    with torch.no_grad():
        out = model(_make_batch())
    _assert_logit_tensor(out)


def test_tcn() -> None:
    from src.models.tcn import TCN1D

    model = TCN1D().eval()
    _assert_under_param_budget(model)
    with torch.no_grad():
        out = model(_make_batch())
    _assert_logit_tensor(out)
