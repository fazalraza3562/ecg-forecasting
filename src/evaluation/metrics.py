"""Pure-numpy/sklearn metrics for the ECG forecasting task.

No file I/O, no model loading, no argparse. Everything in here takes
arrays in and returns scalars or dicts. ``src/evaluation/evaluate.py``
is the orchestration layer that loads checkpoints and calls into these
functions.

The threshold-free helpers are NaN-safe by construction: sklearn's
``roc_auc_score`` raises on single-class inputs, which kills a whole
evaluation pass over a benign edge case (e.g. a val patient with zero
positives). We catch that here and propagate NaN instead.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_threshold_free_metrics(
    y_true: np.ndarray, y_score: np.ndarray,
) -> dict[str, float]:
    """Return AUROC and AUPRC, or NaN for both if y_true is single-class."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def find_threshold_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.95,
) -> float:
    """Smallest threshold whose specificity is at least ``target_spec``.

    Smallest threshold means most sensitive operating point that still
    clears the specificity bar, which is the clinically interesting
    choice — we want to catch as many events as possible without spamming
    alarms above some FPR ceiling.
    """
    fpr, _, thresholds = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    mask = spec >= target_spec
    if not mask.any():
        # No threshold meets the bar; fall back to the most conservative
        # one rather than returning NaN — the caller still gets a usable
        # operating point and the resulting sensitivity will reflect it.
        return float(np.max(thresholds))
    # roc_curve returns thresholds in decreasing order; the highest index
    # where mask is True is the most sensitive operating point that still
    # clears the bar. Using the index (not np.min over values) avoids
    # accidentally selecting the sklearn-prepended sentinel threshold,
    # which trivially achieves spec=1.0 by predicting all-negative.
    idx = np.flatnonzero(mask)[-1]
    return float(thresholds[idx])


def compute_threshold_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float,
) -> dict[str, float]:
    """Confusion-matrix-derived metrics at a fixed decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    # labels=[0, 1] forces a 2x2 even when predictions or labels are
    # all-one-class, so the ravel below never explodes.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Precision is genuinely undefined with no positive predictions —
    # returning 0.0 would silently say "we got everything wrong" when
    # in fact we said nothing. NaN is the honest signal.
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": f1,
    }


def compute_fpr_per_hour(
    fp: int, n_negatives: int, stride_seconds: float,
) -> float:
    """Convert a raw FP count into false alarms per hour of monitoring."""
    if n_negatives <= 0:
        return 0.0
    total_seconds = n_negatives * stride_seconds
    return float(fp * 3600.0 / total_seconds)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Percentile-bootstrap CI around ``metric_fn``.

    Resamples that ``metric_fn`` rejects (ValueError) or returns NaN for
    are dropped and counted via ``n_valid``. If fewer than 10 resamples
    survive we refuse to estimate the CI — the percentile would be
    meaningless — and return NaN for mean/lo/hi while still reporting
    the survivor count.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    samples: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                val = metric_fn(y_true[idx], y_score[idx])
        except ValueError:
            continue
        if val is None or np.isnan(val):
            continue
        samples.append(float(val))

    n_valid = len(samples)
    if n_valid < 10:
        return {
            "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
            "n_valid": n_valid,
        }

    arr = np.asarray(samples)
    alpha = 1.0 - ci
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return {"mean": float(arr.mean()), "lo": lo, "hi": hi, "n_valid": n_valid}


def per_patient_auroc(
    y_true: np.ndarray, y_score: np.ndarray, patient_ids: np.ndarray,
) -> dict[str, float]:
    """AUROC per patient, skipping patients whose labels are single-class."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    patient_ids = np.asarray(patient_ids)
    out: dict[str, float] = {}
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        pid_true = y_true[mask]
        if len(np.unique(pid_true)) < 2:
            continue
        out[str(pid)] = float(roc_auc_score(pid_true, y_score[mask]))
    return out
