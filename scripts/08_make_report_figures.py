"""Render the three report figures from on-disk evaluation artifacts.

Reads ``runs/<model>/<timestamp>/test_metrics.json`` and
``runs/<model>/<timestamp>/test_predictions.npz`` for every model that
has a timestamped run, and writes:

* ``figures/auroc_comparison.png``      — bar chart with bootstrap CIs
* ``figures/calibration.png``           — predicted-probability histograms
* ``figures/prevalence_comparison.png`` — val vs test class balance

No torch, no model loading, no inference. Pure file reads plus
matplotlib so the script runs in seconds on any laptop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
FIGURES_ROOT = REPO_ROOT / "figures"

plt.rcParams["font.family"] = "serif"


def discover_runs() -> dict[str, Path]:
    """Return ``{model_name: latest_timestamped_run_dir}`` for every model."""
    by_model: dict[str, Path] = {}
    for model_dir in sorted(RUNS_ROOT.glob("*")):
        if not model_dir.is_dir():
            continue
        timestamps = sorted(p for p in model_dir.glob("[0-9]*") if p.is_dir())
        if timestamps:
            by_model[model_dir.name] = timestamps[-1]
    return by_model


def _load_metrics(run_dir: Path, split: str) -> dict[str, Any]:
    return json.loads((run_dir / f"{split}_metrics.json").read_text())


def _load_predictions(run_dir: Path, split: str) -> dict[str, np.ndarray]:
    npz = np.load(run_dir / f"{split}_predictions.npz")
    return {k: npz[k] for k in npz.files}


def _split_stats(npz: dict[str, np.ndarray]) -> dict[str, int]:
    labels = npz["labels"]
    return {
        "n_windows": int(len(labels)),
        "n_positives": int((labels == 1).sum()),
        "n_negatives": int((labels == 0).sum()),
        "n_patients": int(len(np.unique(npz["patient_ids"]))),
    }


def make_auroc_comparison(runs: dict[str, Path], test_stats: dict[str, int]) -> Path:
    """Horizontal bar chart of test AUROC per model, with bootstrap CIs."""
    rows = []
    for model, run_dir in runs.items():
        tf = _load_metrics(run_dir, "test")["threshold_free"]
        rows.append({
            "model": model,
            "auroc": float(tf["auroc"]),
            "lo": float(tf["auroc_ci"]["lo"]),
            "hi": float(tf["auroc_ci"]["hi"]),
        })
    rows.sort(key=lambda r: r["auroc"], reverse=True)

    aurocs = np.array([r["auroc"] for r in rows])
    err_lo = aurocs - np.array([r["lo"] for r in rows])
    err_hi = np.array([r["hi"] for r in rows]) - aurocs
    y_pos = np.arange(len(rows))
    colors = plt.cm.tab10(np.arange(len(rows)) % 10)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        y_pos, aurocs, xerr=[err_lo, err_hi],
        capsize=3, color=colors, alpha=0.85,
        edgecolor="black", linewidth=0.5,
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    # Position "chance" label just above the topmost bar, right of the line.
    ax.text(0.505, -0.6, "chance", ha="left", va="center",
            fontsize=9, style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["model"] for r in rows])
    ax.set_xlabel("Test AUROC")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()  # best model on top
    ax.set_title("Test AUROC with 95% bootstrap CI (n=500)")

    fig.text(
        0.5, 0.01,
        f"SDDB test split, {test_stats['n_patients']} patients, "
        f"{test_stats['n_positives']} positives / {test_stats['n_windows']} windows",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out = FIGURES_ROOT / "auroc_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_calibration_plot(runs: dict[str, Path]) -> Path:
    """2x3 grid of probability histograms per model, split by label."""
    model_names = sorted(runs.keys())  # alphabetical so the grid is predictable
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7))
    axes_flat = axes.flatten()
    bins = np.linspace(0.0, 1.0, 31)

    for ax, model in zip(axes_flat, model_names):
        run_dir = runs[model]
        preds = _load_predictions(run_dir, "test")
        labels = preds["labels"]
        probs = preds["probs"]
        auroc = _load_metrics(run_dir, "test")["threshold_free"]["auroc"]

        ax.hist(probs[labels == 0], bins=bins, color="gray",
                alpha=0.5, label="negatives")
        ax.hist(probs[labels == 1], bins=bins, color="crimson",
                alpha=0.7, label="positives")
        ax.set_yscale("log")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{model}  AUROC={auroc:.3f}", fontsize=10)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("count (log)")
        # Real data range is [0, 0.6]; cropping the empty tail past 0.7
        # gives ~30% more usable width without hiding the 0.5 threshold.
        ax.set_xlim(0.0, 0.7)
        ax.legend(loc="upper right", fontsize=8)

    # Hide any leftover axes if fewer than 6 models were discovered.
    for ax in axes_flat[len(model_names):]:
        ax.set_visible(False)

    fig.suptitle("Predicted probability distributions on SDDB test split",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIGURES_ROOT / "calibration.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_prevalence_plot(
    val_stats: dict[str, int], test_stats: dict[str, int],
) -> Path:
    """Two-panel stacked bar chart of val vs test class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    panels = [
        (axes[0], "val", val_stats),
        (axes[1], "test", test_stats),
    ]
    for ax, label, stats in panels:
        neg, pos = stats["n_negatives"], stats["n_positives"]
        total = neg + pos
        prev = 100.0 * pos / max(total, 1)
        ax.bar([0], [neg], color="gray", label="negatives",
               edgecolor="black", linewidth=0.5)
        ax.bar([0], [pos], bottom=[neg], color="crimson", label="positives",
               edgecolor="black", linewidth=0.5)
        # Annotation sits just above the stacked bar.
        ax.text(0, total + 0.025 * total, f"{prev:.2f}% positive",
                ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{label} split (n={total:,})")
        ax.set_xticks([])
        ax.set_ylabel("windows")
        ax.set_ylim(0, total * 1.15)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Class prevalence: val vs test splits", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = FIGURES_ROOT / "prevalence_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    runs = discover_runs()
    if not runs:
        raise SystemExit("no run directories found under runs/*/[0-9]*/")
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    # The test/val splits are shared across models so we only need to
    # read predictions once for the shared-stats numbers.
    any_run = next(iter(runs.values()))
    test_stats = _split_stats(_load_predictions(any_run, "test"))
    val_stats = _split_stats(_load_predictions(any_run, "val"))

    print(f"wrote {make_auroc_comparison(runs, test_stats)}")
    print(f"wrote {make_calibration_plot(runs)}")
    print(f"wrote {make_prevalence_plot(val_stats, test_stats)}")


if __name__ == "__main__":
    main()
