"""Cosine learning-rate schedule with linear warmup.

A single helper that wraps the warmup + cosine-decay schedule everyone
re-implements three times a year. It returns a ``LambdaLR``, so the
returned object obeys the standard PyTorch scheduler API and slots
straight into the trainer.
"""
from __future__ import annotations

import math

import torch


def make_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from 0 to base LR, then cosine decay down to ``min_lr_ratio * base_lr``.

    The schedule is per-step, not per-epoch: call ``scheduler.step()`` after
    every ``optimizer.step()``, not at the end of each epoch.

    Args:
        optimizer: Optimizer whose param-group learning rates are the base LRs.
        num_warmup_steps: Linear-warmup duration in optimizer steps. ``0``
            disables warmup and the schedule starts at base LR.
        num_training_steps: Total optimizer steps over the whole run. Steps
            beyond this are clamped at ``min_lr_ratio * base_lr``.
        min_lr_ratio: Fraction of base LR at the end of the cosine decay.
            ``0.0`` decays all the way to zero.

    Returns:
        A ``LambdaLR`` whose lambda implements the schedule.
    """
    if num_warmup_steps < 0:
        raise ValueError(f"num_warmup_steps must be non-negative, got {num_warmup_steps}")
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be positive, got {num_training_steps}")
    if num_warmup_steps >= num_training_steps:
        raise ValueError(
            f"num_warmup_steps ({num_warmup_steps}) must be strictly less than "
            f"num_training_steps ({num_training_steps}); otherwise there is no "
            f"decay region and the schedule is undefined."
        )
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    decay_steps = num_training_steps - num_warmup_steps

    def lr_lambda(current_step: int) -> float:
        # Linear ramp up to 1.0 at step == num_warmup_steps. The max(1, .)
        # is a divide-by-zero guard for the zero-warmup case, even though
        # the branch is unreachable when num_warmup_steps == 0.
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        # Clamp at the floor once we've burned through the budget rather
        # than letting cosine wrap around to 1 again.
        if current_step >= num_training_steps:
            return min_lr_ratio
        # Cosine decay: progress in [0, 1), cosine factor in (0, 1].
        progress = (current_step - num_warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
