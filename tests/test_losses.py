"""Sanity tests for the imbalanced-class losses.

These tests pin down three load-bearing properties:

* The focal weighting reduces to vanilla BCE at gamma=0, so anything
  downstream that swaps focal for BCE for an ablation is a one-parameter
  change.
* The focal weighting actually down-weights easy examples by a meaningful
  amount, not just a sub-percent rounding adjustment.
* Both losses stay finite on logits at +/-100, which is the practical
  range we hit on confident predictions partway through training.

The pos-weight helper has its own test because the all-negative-batch
divide-by-zero is a real pitfall — we hit it once when a stride change
left one validation patient with no positives.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.training.losses import (
    FocalLoss,
    compute_pos_weight_from_labels,
)


def test_focal_loss_matches_bce_at_gamma_zero() -> None:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(32, generator=g)
    targets = (torch.rand(32, generator=g) > 0.5).float()

    focal = FocalLoss(gamma=0.0, alpha=None)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

    assert torch.allclose(focal, bce, atol=1e-6), (
        f"focal at gamma=0 should equal BCE; got focal={focal.item()} bce={bce.item()}"
    )


def test_focal_loss_downweights_easy_examples() -> None:
    # A logit of +10 with target 1 is the textbook "easy positive": sigmoid is
    # essentially 1, BCE is tiny, focal should be tinier still.
    logits = torch.tensor([10.0])
    targets = torch.tensor([1.0])

    focal = FocalLoss(gamma=2.0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

    assert focal.item() < 0.1 * bce.item(), (
        f"focal should heavily down-weight easy examples; "
        f"got focal={focal.item():.3e} vs 0.1 * bce={0.1 * bce.item():.3e}"
    )


def test_focal_loss_handles_extreme_logits() -> None:
    # Either sign of saturated logit. The naive sigmoid-then-log form goes
    # to log(0) here; the BCE-with-logits form stays finite.
    logits = torch.tensor([100.0, -100.0])
    targets = torch.tensor([1.0, 0.0])

    loss = FocalLoss(gamma=2.0)(logits, targets)

    assert torch.isfinite(loss).all(), f"focal loss is non-finite: {loss}"


def test_compute_pos_weight() -> None:
    labels = np.concatenate([np.ones(10), np.zeros(990)])
    assert compute_pos_weight_from_labels(labels) == 99.0

    # All-negative split: without the max(., 1) guard this would be inf.
    # The guard sends pos_weight -> n_negatives, which is finite and large
    # enough to keep BCE honest if any positive ever does show up.
    all_negatives = np.zeros(500)
    assert compute_pos_weight_from_labels(all_negatives) == 500.0
