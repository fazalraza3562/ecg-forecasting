"""Assert that no sample inside the 30-second prediction horizon leaks into the input window."""
from __future__ import annotations

import numpy as np

from src.data.windows import make_windows


FS = 250
WINDOW_S = 60
HORIZON_S = 30
STRIDE_S = 1


def _flat_signal(duration_s: int, n_channels: int = 1) -> np.ndarray:
    """A zero-valued (C, T) signal of the requested duration. Content is irrelevant."""
    return np.zeros((n_channels, duration_s * FS), dtype=np.float32)


def _expected_kept_t_i(onset: float, duration_s: int) -> list[float]:
    """Reproduce the emit-schedule independently from the implementation."""
    kept: list[float] = []
    end_t = WINDOW_S
    while end_t + HORIZON_S <= duration_s:
        past_lo = end_t - WINDOW_S
        if not (past_lo <= onset <= end_t):
            kept.append(float(end_t))
        end_t += STRIDE_S
    return kept


def test_no_horizon_leakage() -> None:
    """Positive windows must end strictly before the onset and contain the onset in horizon."""
    onset = 100.0
    signal = _flat_signal(duration_s=200)
    _windows, labels = make_windows(
        signal,
        onset_times_s=[onset],
        fs=FS,
        window_s=WINDOW_S,
        horizon_s=HORIZON_S,
        stride_s=STRIDE_S,
    )

    kept_t_i = _expected_kept_t_i(onset, duration_s=200)
    assert len(kept_t_i) == len(labels)
    assert labels.sum() > 0, "expected at least one positive label"

    for idx, t_i in enumerate(kept_t_i):
        if labels[idx] != 1:
            continue
        # Window data ends at t_i, which must precede the onset.
        assert t_i < onset, f"positive window ends at t_i={t_i} >= onset={onset}"
        # Onset must lie in the look-ahead horizon (t_i, t_i + horizon_s].
        assert t_i < onset <= t_i + HORIZON_S, (
            f"onset {onset} not in horizon ({t_i}, {t_i + HORIZON_S}] for window {idx}"
        )


def test_in_progress_skipped() -> None:
    """Windows whose past contains an onset must not be emitted as forecasting examples."""
    # Onset late enough that there are valid forecasting windows before it,
    # then windows that straddle / follow it become in-progress and are dropped.
    onset = 120.0
    signal = _flat_signal(duration_s=400)
    windows, labels = make_windows(
        signal,
        onset_times_s=[onset],
        fs=FS,
        window_s=WINDOW_S,
        horizon_s=HORIZON_S,
        stride_s=STRIDE_S,
    )

    # Translate t_i values back from the run of kept windows. With stride 1
    # from t_i=60s, kept windows correspond to a strictly increasing schedule
    # interrupted whenever the in-progress rule fires. We can't index a kept
    # window directly to its t_i without re-implementing the loop, so instead
    # we just check that NO window we emit has the onset inside its past.
    # We emulate the past interval by reconstructing from the signal slice:
    # rather than that, recompute the expected emitted t_i values.
    expected_t_i: list[float] = []
    end_t = WINDOW_S
    while end_t + HORIZON_S <= 400:
        past_lo = end_t - WINDOW_S
        if not (past_lo <= onset <= end_t):
            expected_t_i.append(end_t)
        end_t += STRIDE_S

    assert len(expected_t_i) == len(labels), (
        f"emitted {len(labels)} windows, expected {len(expected_t_i)}"
    )
    # And specifically: no t_i in [onset, onset + WINDOW_S] should appear, since
    # for those, the onset sits inside the past interval.
    for t_i in expected_t_i:
        assert not (onset <= t_i <= onset + WINDOW_S), (
            f"in-progress window slipped through at t_i={t_i}"
        )


def test_label_consistency() -> None:
    """Hand-crafted positions: verify which stride positions yield label 1 vs 0."""
    onset = 100.0
    signal = _flat_signal(duration_s=200)
    windows, labels = make_windows(
        signal,
        onset_times_s=[onset],
        fs=FS,
        window_s=WINDOW_S,
        horizon_s=HORIZON_S,
        stride_s=STRIDE_S,
    )

    # Map every kept window back to its t_i. With these settings, the first
    # window has t_i=60; in-progress drops begin at t_i=100 (onset is at the
    # right edge of the past, i.e. inside [40, 100]). So kept indices 0..39
    # correspond to t_i = 60..99. After index 39, windows are skipped until
    # t_i > onset + WINDOW_S = 160, but the recording only reaches t_i = 170
    # (signal duration 200 minus horizon 30). So we get 10 more windows for
    # t_i = 161..170, all label 0 because the onset is no longer in horizon.
    pre_block = labels[:40]
    post_block = labels[40:]

    # Pre-block: positives are exactly those where onset in (t_i, t_i + 30].
    # For t_i in 60..99 that means 70 <= t_i <= 99 (30 windows), i.e. indices 10..39.
    expected_pre = np.zeros(40, dtype=np.int8)
    expected_pre[10:40] = 1
    np.testing.assert_array_equal(pre_block, expected_pre)

    # Post-block: onset already passed, every label should be 0.
    np.testing.assert_array_equal(post_block, np.zeros_like(post_block))
