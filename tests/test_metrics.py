"""Smoke tests for src/evaluation/metrics.py.

Targets the corner cases that actually bit us during prototyping:
single-class validation splits, all-zero predictions early in training,
and bootstrap resamples that happen to draw all-one-class subsets.
The arithmetic checks use small hand-computable inputs so a failure
points at a specific bug rather than at "the numbers moved".
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    bootstrap_ci,
    compute_fpr_per_hour,
    compute_threshold_free_metrics,
    compute_threshold_metrics,
    find_threshold_at_specificity,
    per_patient_auroc,
)


def test_threshold_free_metrics_known_auroc() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    out = compute_threshold_free_metrics(y_true, y_score)
    # Pairs (pos > neg): (0.35>0.1)=1, (0.35>0.4)=0, (0.8>0.1)=1, (0.8>0.4)=1.
    # AUROC = 3 / (n_pos * n_neg) = 3/4 = 0.75.
    assert abs(out["auroc"] - 0.75) < 1e-6, f"got AUROC={out['auroc']}"
    assert out["auprc"] > 0.7  # ~0.833, leave room for tiny numerical drift


def test_threshold_free_metrics_single_class_returns_nan() -> None:
    y_true = np.zeros(10)
    y_score = np.linspace(0, 1, 10)
    out = compute_threshold_free_metrics(y_true, y_score)
    assert np.isnan(out["auroc"]), f"expected NaN AUROC, got {out['auroc']}"
    assert np.isnan(out["auprc"]), f"expected NaN AUPRC, got {out['auprc']}"


def test_find_threshold_at_specificity_round_trip() -> None:
    # Mostly-separable data so a meaningful threshold exists.
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(80), np.ones(20)])
    y_score = np.concatenate([
        rng.normal(0.2, 0.1, size=80),
        rng.normal(0.7, 0.1, size=20),
    ])
    thr = find_threshold_at_specificity(y_true, y_score, target_spec=0.95)
    metrics = compute_threshold_metrics(y_true, y_score, thr)
    assert metrics["specificity"] >= 0.95, (
        f"specificity {metrics['specificity']} below target 0.95 at thr={thr}"
    )
    # Guards against returning the sklearn sentinel threshold, which
    # trivially clears any specificity bar with sensitivity=0.
    assert metrics["sensitivity"] > 0.5, (
        f"sensitivity {metrics['sensitivity']} suggests degenerate threshold thr={thr}"
    )


def test_threshold_metrics_all_zero_predictions() -> None:
    # Threshold above every score => y_pred is all zeros.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    out = compute_threshold_metrics(y_true, y_score, threshold=1.5)
    assert out["tp"] == 0 and out["fp"] == 0
    assert out["fn"] == 2 and out["tn"] == 2
    assert out["sensitivity"] == 0.0
    assert out["specificity"] == 1.0
    assert out["f1"] == 0.0
    # Precision is NaN (no positive predictions made), not 0.
    assert np.isnan(out["precision"])


def test_fpr_per_hour_known_value() -> None:
    # 10 false positives over 1000 negative windows at 30 s stride =>
    # 10 * 3600 / (1000 * 30) = 1.2 alarms/hour.
    rate = compute_fpr_per_hour(fp=10, n_negatives=1000, stride_seconds=30.0)
    assert abs(rate - 1.2) < 1e-9, f"got {rate}/h"

    # Empty population is the edge case the caller most commonly hits.
    assert compute_fpr_per_hour(fp=0, n_negatives=0, stride_seconds=30.0) == 0.0


def test_bootstrap_ci_on_known_auroc() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    def auroc_fn(yt: np.ndarray, ys: np.ndarray) -> float:
        return compute_threshold_free_metrics(yt, ys)["auroc"]

    out = bootstrap_ci(y_true, y_score, auroc_fn, n_resamples=500, seed=42)
    # With n=4 and ~50/50 prevalence, ~12.5% of resamples are single-class.
    # We expect ~430 of 500 to survive; the threshold of 400 leaves slack.
    assert out["n_valid"] > 400, f"n_valid={out['n_valid']} too low"
    assert abs(out["mean"] - 0.75) < 0.05, f"mean drifted: {out['mean']}"
    # AUROC is bounded in [0, 1] and the CI must bracket the mean.
    assert 0.0 <= out["lo"] <= out["mean"] <= out["hi"] <= 1.0, (
        f"CI order/bounds broken: lo={out['lo']} mean={out['mean']} hi={out['hi']}"
    )


def test_per_patient_auroc_skips_single_class_patient() -> None:
    patient_ids = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    # Patient A has both classes; patient B is all positive.
    y_true = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.4, 0.5, 0.6])
    out = per_patient_auroc(y_true, y_score, patient_ids)
    assert "A" in out
    assert "B" not in out
    assert out["A"] == 1.0  # patient A is perfectly separable
