"""Deterministic seeding helper used by every training and evaluation entry point."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.

    Also flips cuDNN into deterministic mode. The benchmark flag is turned off
    in the same call because leaving it on lets cuDNN pick nondeterministic
    algorithms even when deterministic=True is set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
