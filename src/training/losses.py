"""Focal loss and class-weighted BCE for the imbalanced positive class.

The SDDB positive rate is well under 1%, so vanilla BCE drives the model
toward the trivial "always predict negative" optimum. Two standard fixes
live in this module:

* :class:`FocalLoss` — Lin et al. 2017. Down-weights examples the model is
  already confident-and-correct on by a factor of ``(1 - p_t)**gamma``,
  which lets the gradient focus on the hard positives that drive AUPRC.
* :class:`WeightedBCELoss` — vanilla BCE with a per-positive scaling factor.
  Cheaper and more interpretable than focal; useful as a baseline so we can
  attribute any improvement to focal's hardness-weighting rather than just
  to seeing positives more often.

The focal implementation deliberately routes through
``binary_cross_entropy_with_logits`` rather than computing ``sigmoid``
then ``log``. With logits in the ``+/-50`` range, the naive form goes to
0 or 1 in float32 and the log blows up; the logits-aware form stays
finite. We rederive ``p_t`` from the BCE output via ``exp(-bce)``, which
is numerically safe because ``bce`` is bounded below by 0.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss computed from logits in a numerically stable way.

    Args:
        gamma: Focusing parameter. ``gamma=0`` recovers vanilla BCE; the
            paper's default ``gamma=2`` is what we use everywhere.
        alpha: Optional class balance term in ``(0, 1)``. ``alpha`` is the
            positive-class weight and ``1 - alpha`` the negative-class
            weight, applied per-example. ``None`` skips alpha entirely.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if alpha is not None and not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1) or None, got {alpha}")
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE-with-logits computes -log(p_t) directly from the logits using
        # the log-sum-exp trick, so it survives logits at +/-100. We then
        # recover p_t = exp(-bce) for the focal weight; exp of a negative
        # number is in (0, 1] regardless of how large bce gets.
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        return loss.mean()


class WeightedBCELoss(nn.Module):
    """BCE-with-logits with a fixed positive-class weight.

    The weight is registered as a buffer, not a plain attribute, so
    ``.to(device)`` and ``.cuda()`` move it along with the module. That
    avoids the silent-CPU-tensor bug that bites people who store the
    weight as a raw ``self.pos_weight = torch.tensor(...)``.
    """

    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        if pos_weight <= 0:
            raise ValueError(f"pos_weight must be positive, got {pos_weight}")
        self.register_buffer(
            "pos_weight",
            torch.tensor([float(pos_weight)], dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="mean",
        )


def compute_pos_weight_from_labels(labels: np.ndarray) -> float:
    """Return ``n_negatives / max(n_positives, 1)``.

    The ``max(., 1)`` guard handles the degenerate case where a split
    contains no positives — without it we'd return ``inf`` and silently
    poison whatever loss this gets fed to. Returning ``len(labels)`` in
    that case is a sensible (if conservative) upper bound that keeps the
    loss finite while still telling the optimizer "positives, if any,
    matter a lot".
    """
    labels = np.asarray(labels)
    n_pos = int((labels > 0).sum())
    n_neg = int(labels.size - n_pos)
    return n_neg / max(n_pos, 1)
