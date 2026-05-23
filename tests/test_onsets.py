"""Tests for the SDDB onset rule.

SDDB's main ``.atr`` files only carry beat-level symbols, so we infer VT
onset from runs of consecutive ``V`` beats. The rule has two knobs that
matter — minimum run length, and a merge window that collapses runs that
belong to the same arrhythmic episode. This file pins both down.
"""
from __future__ import annotations

import numpy as np

from src.data.preprocess import _ventricular_onsets_from_annotation


FS = 250.0
MERGE_S = 60.0


def _v_run(start_sample: int, count: int, gap_samples: int = 250) -> list[int]:
    """A list of ``count`` sample indices spaced ``gap_samples`` apart."""
    return [start_sample + i * gap_samples for i in range(count)]


def test_sddb_merges_within_window_and_keeps_distant_episodes() -> None:
    """The headline scenario from the spec: four candidate episodes -> two onsets.

    Run A: 5 V's at t=100s (kept).
    Run B: 4 V's at t=130s (within 60s of A -> merged into A).
    Run C: 3 V's at t=200s (100s past A -> separate episode).
    Run D: 2 isolated V's at t=300s (below the 3-beat threshold -> ignored).
    """
    a_samples = _v_run(start_sample=int(100 * FS), count=5)
    b_samples = _v_run(start_sample=int(130 * FS), count=4)
    c_samples = _v_run(start_sample=int(200 * FS), count=3)
    d_samples = _v_run(start_sample=int(300 * FS), count=2)

    # Separator beats between runs so the runs don't merge into one giant run.
    # ``N`` is the natural choice — a sinus beat between bursts of V's.
    samples: list[int] = []
    symbols: list[str] = []
    for run in (a_samples, b_samples, c_samples, d_samples):
        samples.extend(run)
        symbols.extend(["V"] * len(run))
        # Insert a separating N beat 0.5 s after the last V of the run.
        samples.append(run[-1] + int(0.5 * FS))
        symbols.append("N")

    onsets = _ventricular_onsets_from_annotation(
        samples=np.asarray(samples, dtype=np.int64),
        symbols=symbols,
        aux_notes=[""] * len(symbols),
        fs=FS,
        db_name="sddb",
        merge_window_s=MERGE_S,
    )

    assert onsets == [100.0, 200.0], f"expected [100.0, 200.0], got {onsets}"


def test_sddb_ignores_runs_shorter_than_three() -> None:
    """Singletons (PVCs) and couplets must not produce onsets."""
    samples = [0, 250, 1000, 2000, 2250]                  # 2 V's, then 1 V, then 2 V's
    symbols = ["V", "V", "N", "V", "V"]
    onsets = _ventricular_onsets_from_annotation(
        samples=np.asarray(samples, dtype=np.int64),
        symbols=symbols,
        aux_notes=[""] * len(symbols),
        fs=FS,
        db_name="sddb",
        merge_window_s=MERGE_S,
    )
    assert onsets == []


def test_sddb_run_at_end_of_record_is_detected() -> None:
    """A run that extends to the final annotation must still emit its onset."""
    start = int(50 * FS)
    samples = _v_run(start_sample=start, count=4)
    symbols = ["V"] * 4
    onsets = _ventricular_onsets_from_annotation(
        samples=np.asarray(samples, dtype=np.int64),
        symbols=symbols,
        aux_notes=[""] * len(symbols),
        fs=FS,
        db_name="sddb",
        merge_window_s=MERGE_S,
    )
    assert onsets == [50.0]


def test_sddb_merge_chain_compares_to_last_accepted_not_last_seen() -> None:
    """Onsets at 100, 130, 170 with a 60s window collapse to a single onset.

    130 merges into 100 because it's within 60s of 100. 170 is 70s past 130
    but only 70s past 100 as well — but the merge rule compares against the
    last *accepted* onset (100), so 170 - 100 = 70 > 60, and 170 is kept.
    """
    samples: list[int] = []
    symbols: list[str] = []
    for t in (100.0, 130.0, 170.0):
        run = _v_run(start_sample=int(t * FS), count=3)
        samples.extend(run)
        symbols.extend(["V"] * len(run))
        samples.append(run[-1] + int(0.5 * FS))
        symbols.append("N")

    onsets = _ventricular_onsets_from_annotation(
        samples=np.asarray(samples, dtype=np.int64),
        symbols=symbols,
        aux_notes=[""] * len(symbols),
        fs=FS,
        db_name="sddb",
        merge_window_s=MERGE_S,
    )
    assert onsets == [100.0, 170.0]


def test_mitdb_rule_still_uses_aux_notes_and_does_not_merge() -> None:
    """Sanity check: the MIT-BIH branch is untouched by the SDDB-specific rule."""
    samples = [int(50 * FS), int(80 * FS), int(120 * FS)]
    symbols = ["+", "+", "+"]
    aux_notes = ["(VT", "(N", "(VFL"]
    onsets = _ventricular_onsets_from_annotation(
        samples=np.asarray(samples, dtype=np.int64),
        symbols=symbols,
        aux_notes=aux_notes,
        fs=FS,
        db_name="mitdb",
        merge_window_s=MERGE_S,
    )
    assert onsets == [50.0, 120.0]
