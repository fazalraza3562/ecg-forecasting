"""Build 60-second input windows paired with a 30-second look-ahead label, no leakage.

The forecasting framing is the heart of this project. Each window is a tuple
``(past, future_label)`` where:

    past   = signal in the closed interval [t_i - window_s, t_i]
    future = the open-left / closed-right interval (t_i, t_i + horizon_s]
    label  = 1 iff at least one VT/VF onset falls into ``future``

Two invariants must hold and are checked by ``tests/test_windows.py``:

1. No sample from ``future`` ever leaks into ``past``. The window data ends at
   ``t_i`` and the horizon starts strictly after ``t_i``.
2. Windows that are themselves *inside* an arrhythmia are dropped. We do not
   ask the model to forecast something that has already begun; that is a
   different (and easier) detection problem. A window is considered
   "in-progress" if any onset falls within its past interval
   ``[t_i - window_s, t_i]``.
"""
from __future__ import annotations

import numpy as np


def make_windows(
    signal: np.ndarray,
    onset_times_s: list[float],
    fs: int,
    window_s: int,
    horizon_s: int,
    stride_s: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of ``window_s`` seconds across ``signal`` with given stride.

    The returned windows are forecasting examples. The signal inside each
    window ends at some time ``t_i``; the label is 1 if any onset falls in
    ``(t_i, t_i + horizon_s]``, otherwise 0. Windows whose own past already
    contains an onset are skipped — they are mid-arrhythmia and not a
    legitimate forecasting target.

    Args:
        signal: ECG samples shaped ``(C, T_total)``; C channels, T_total samples.
        onset_times_s: VT/VF onset times in seconds since the record start.
        fs: Sampling rate of ``signal`` in Hz.
        window_s: Length of the input window in seconds.
        horizon_s: Look-ahead length in seconds.
        stride_s: Step between successive window end-times in seconds.

    Returns:
        windows: float32 array of shape ``(N, C, window_s * fs)``.
        labels:  int8 array of shape ``(N,)`` with values in ``{0, 1}``.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2-D (C, T); got shape {signal.shape}")
    if window_s <= 0 or horizon_s <= 0 or stride_s <= 0:
        raise ValueError("window_s, horizon_s, stride_s must all be positive")

    n_channels, n_samples = signal.shape
    window_samples = window_s * fs
    stride_samples = stride_s * fs

    onsets = np.asarray(onset_times_s, dtype=np.float64)

    windows_out: list[np.ndarray] = []
    labels_out: list[int] = []

    end_idx = window_samples
    last_end_idx = n_samples - horizon_s * fs

    while end_idx <= last_end_idx:
        start_idx = end_idx - window_samples
        t_i = end_idx / fs

        # In-progress check: any onset inside the past interval [t_i - window_s, t_i]
        # means the recording is mid-arrhythmia, not a forecasting opportunity.
        if onsets.size > 0:
            past_lo = t_i - window_s
            in_past = (onsets >= past_lo) & (onsets <= t_i)
            if np.any(in_past):
                end_idx += stride_samples
                continue

            # Forecast horizon (t_i, t_i + horizon_s]: strictly after the window.
            in_horizon = (onsets > t_i) & (onsets <= t_i + horizon_s)
            label = int(np.any(in_horizon))
        else:
            label = 0

        windows_out.append(signal[:, start_idx:end_idx].astype(np.float32, copy=False))
        labels_out.append(label)
        end_idx += stride_samples

    if not windows_out:
        return (
            np.zeros((0, n_channels, window_samples), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
        )

    return np.stack(windows_out, axis=0), np.asarray(labels_out, dtype=np.int8)
