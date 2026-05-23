"""Smoke tests for the cosine-with-warmup schedule.

The schedule has three regimes (warmup ramp, cosine decay, post-training
floor) and one degenerate case (zero warmup). These tests pin down the
hand-offs between them: ramp starts at 0, hits base LR exactly at the
end of warmup, decays close to the floor by the last step, and starts
at base LR immediately when warmup is disabled.
"""
from __future__ import annotations

import torch

from src.training.scheduler import make_cosine_schedule_with_warmup


BASE_LR = 1e-3


def _make_optimizer(lr: float = BASE_LR) -> torch.optim.Optimizer:
    # A single trainable scalar is enough — the scheduler doesn't care
    # about the parameters, only the optimizer's param-group LR. We give
    # it a zero gradient so the dummy optimizer.step() in the tests is
    # a real no-op and PyTorch doesn't print the "scheduler before
    # optimizer" warning that fires on a never-stepped optimizer.
    param = torch.nn.Parameter(torch.zeros(1))
    param.grad = torch.zeros_like(param)
    return torch.optim.SGD([param], lr=lr)


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def test_warmup_starts_at_zero_or_near_zero() -> None:
    opt = _make_optimizer()
    make_cosine_schedule_with_warmup(opt, num_warmup_steps=100, num_training_steps=1000)
    # LambdaLR applies the multiplier at construction time, so we read
    # the LR straight off the optimizer without stepping.
    assert _current_lr(opt) == 0.0, f"expected LR=0 at step 0, got {_current_lr(opt)}"


def test_warmup_reaches_base_lr_at_warmup_end() -> None:
    opt = _make_optimizer()
    sched = make_cosine_schedule_with_warmup(
        opt, num_warmup_steps=100, num_training_steps=1000,
    )
    # Step 100 times to reach the end of warmup. The dummy optimizer.step()
    # before each scheduler.step() mirrors the trainer's call order.
    for _ in range(100):
        opt.step()
        sched.step()
    assert abs(_current_lr(opt) - BASE_LR) < 1e-6, (
        f"expected LR={BASE_LR} at end of warmup, got {_current_lr(opt)}"
    )


def test_cosine_reaches_min_at_end() -> None:
    min_ratio = 0.1
    opt = _make_optimizer()
    sched = make_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=100,
        num_training_steps=1000,
        min_lr_ratio=min_ratio,
    )
    # Step to num_training_steps - 1: cosine is continuous so we won't hit
    # the floor exactly, but we should be within a small fraction of base LR.
    for _ in range(999):
        opt.step()
        sched.step()
    expected_min = min_ratio * BASE_LR
    assert abs(_current_lr(opt) - expected_min) < 1e-3 * BASE_LR, (
        f"expected LR ~= {expected_min} at end of decay, got {_current_lr(opt)}"
    )


def test_zero_warmup_starts_at_base_lr() -> None:
    opt = _make_optimizer()
    make_cosine_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=1000,
    )
    assert abs(_current_lr(opt) - BASE_LR) < 1e-6, (
        f"expected LR={BASE_LR} at step 0 with zero warmup, got {_current_lr(opt)}"
    )
