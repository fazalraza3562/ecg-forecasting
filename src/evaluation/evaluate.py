"""Per-model evaluation CLI, single dataset or cross-dataset.

Loads a trained checkpoint and runs inference on either an SDDB split
(val / test, requires ``data/processed/sddb/split.json``) or the full
preprocessed MIT-BIH / INCART set. Writes threshold-free, fixed-threshold,
and val-tuned-threshold metrics next to the checkpoint. With ``--all``,
the same flow runs over every architecture in ``MODELS`` and produces a
top-level comparison CSV plus Markdown summary.

The val-tuned threshold is always derived from the SDDB val predictions
on disk (``<run_dir>/val_predictions.npz``). For cross-dataset eval this
is the only honest choice — we cannot tune a threshold on the held-out
dataset itself without leaking labels.

Three rules the script enforces structurally:

* No threshold tuning on the same labels we report. Val -> threshold ->
  test (or val -> threshold -> mitdb/incartdb), never the other way.
* No CPU autocast. AMP is only entered when the resolved device is
  CUDA; CPU runs stay in fp32 because PyTorch's CPU autocast path has
  surprised us once already.
* No DataLoader worker processes. The data path contains spaces and
  ``num_workers > 0`` hits a known multiprocessing pickling fragility
  on some platforms.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import ECGWindowDataset
from src.evaluation.metrics import (
    bootstrap_ci,
    compute_fpr_per_hour,
    compute_threshold_free_metrics,
    compute_threshold_metrics,
    find_threshold_at_specificity,
    per_patient_auroc,
)
from src.training.train import MODELS, _load_merged_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
DATASETS = ("sddb", "mitdb", "incartdb")

# Clip logits before sigmoid: float32 sigmoid overflows around +/-89, and
# our trained models occasionally produce a logit in the +/-50 range on
# unusual inputs. We clip the input, not the output, so the probability
# itself isn't artificially flattened.
LOGIT_CLIP = 50.0


def _dataset_paths(dataset: str) -> tuple[Path, Path | None]:
    """Return (windows.npz, split.json or None) for a dataset name.

    Only SDDB has a split.json — MIT-BIH and INCART are evaluated as
    single held-out sets, so there's no split file to read.
    """
    base = REPO_ROOT / "data" / "processed" / dataset
    npz = base / "windows.npz"
    split_json = base / "split.json" if dataset == "sddb" else None
    return npz, split_json


def _artifact_prefix(dataset: str, split: str | None) -> str:
    """Filename prefix for per-model artifacts.

    SDDB writes per-split (``val_metrics.json`` / ``test_metrics.json``);
    cross-dataset evals write per-dataset (``mitdb_metrics.json``).
    """
    if dataset == "sddb":
        assert split is not None, "SDDB requires a split"
        return split
    return dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained ECG forecasting models.")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model", choices=sorted(MODELS.keys()),
        help="Which trained model to evaluate.",
    )
    target.add_argument(
        "--all", action="store_true",
        help="Evaluate every model in MODELS and write a comparison table.",
    )
    p.add_argument("--dataset", default="sddb", choices=list(DATASETS))
    # --split is required only when --dataset sddb; validated in main().
    p.add_argument("--split", choices=["val", "test"], default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _latest_run_dir(model_name: str) -> Path:
    """Latest timestamped run dir under runs/<model>/ that has a best.pt.

    Filter to subdirs that actually contain a checkpoint so a sibling
    figures/ or scratch/ folder created later by downstream notebooks
    can't be mistaken for the trained run.
    """
    base = RUNS_ROOT / model_name
    candidates = sorted(
        p for p in base.glob("*")
        if p.is_dir() and (p / "best.pt").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no run directory with best.pt under {base}")
    return candidates[-1]


def _load_checkpoint(
    ckpt_path: Path, device: torch.device, logger: logging.Logger,
) -> Any:
    """Load a checkpoint, preferring ``weights_only=True``."""
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as exc:
        logger.warning(
            "weights_only=True failed for %s (%s); retrying with weights_only=False",
            ckpt_path, exc,
        )
        return torch.load(ckpt_path, map_location=device, weights_only=False)


def _model_state_dict(state: Any) -> dict[str, torch.Tensor]:
    """Pull the state_dict out of a wrapped checkpoint, or pass through a bare one."""
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def _build_dataset(dataset: str, split: str | None) -> ECGWindowDataset:
    """Construct an ECGWindowDataset for the requested (dataset, split).

    SDDB selects records by reading ``split.json`` and intersecting with
    ``record_names``. MIT-BIH / INCART evaluate the whole npz — there is
    no per-record split for them.
    """
    npz_path, split_json = _dataset_paths(dataset)
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found; run preprocessing first.")

    if dataset != "sddb":
        return ECGWindowDataset(npz_path, indices=None)

    assert split is not None and split_json is not None
    if not split_json.exists():
        raise FileNotFoundError(f"{split_json} not found.")

    with split_json.open() as f:
        split_dict = json.load(f)
    if split not in split_dict:
        raise KeyError(f"split '{split}' not present in {split_json}")

    record_names_all = np.load(npz_path)["record_names"]
    # split.json stores entries as "<dataset>/<record>" (e.g. "sddb/35") to
    # leave room for cross-dataset splits later; the npz only carries the
    # bare record ID. Strip the dataset prefix before matching.
    split_records = np.asarray([r.split("/", 1)[-1] for r in split_dict[split]])
    indices = np.flatnonzero(np.isin(record_names_all, split_records))
    if indices.size == 0:
        raise RuntimeError(f"no windows match split '{split}'; check split.json")

    return ECGWindowDataset(npz_path, indices=indices)


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) as float32 numpy arrays in dataset order."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(windows)
        else:
            logits = model(windows)
        all_logits.append(logits.detach().cpu().float().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)


def _sigmoid_from_logits(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)
    return 1.0 / (1.0 + np.exp(-clipped))


def _auroc_metric(yt: np.ndarray, ys: np.ndarray) -> float:
    return float(roc_auc_score(yt, ys))


def _auprc_metric(yt: np.ndarray, ys: np.ndarray) -> float:
    # AUPRC doesn't raise on single-class y_true the way AUROC does, so
    # the bootstrap wouldn't drop those resamples by itself. We force a
    # ValueError here so bootstrap_ci's accounting reflects reality.
    if len(np.unique(yt)) < 2:
        raise ValueError("single-class resample")
    return float(average_precision_score(yt, ys))


def _read_meta_field(run_dir: Path, key: str) -> Any:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text()).get(key)
    except Exception:
        return None


def _read_param_count(run_dir: Path, model_name: str) -> int:
    n = _read_meta_field(run_dir, "n_params")
    if isinstance(n, int):
        return n
    return sum(p.numel() for p in MODELS[model_name]().parameters())


def _build_per_patient_table(
    dataset: ECGWindowDataset, probs: np.ndarray,
) -> pd.DataFrame:
    pid_array = np.asarray(dataset.patient_ids)
    labels = dataset.labels
    auroc_by_pid = per_patient_auroc(labels, probs, pid_array)

    rows: list[dict[str, Any]] = []
    for pid in np.unique(pid_array):
        mask = pid_array == pid
        rows.append({
            "patient_id": str(pid),
            "n_windows": int(mask.sum()),
            "n_positives": int(labels[mask].sum()),
            # Single-class patients are skipped by per_patient_auroc; we
            # surface NaN so the row still appears in the CSV — useful to
            # see which patients couldn't be scored and why.
            "auroc": auroc_by_pid.get(str(pid), float("nan")),
        })
    return pd.DataFrame(rows)


def _evaluate_single(
    model_name: str,
    dataset: str,
    split: str | None,
    device: torch.device,
    logger: logging.Logger,
    seed: int,
) -> dict[str, Any]:
    """Run inference and metrics for one model on one (dataset, split).

    Writes ``<prefix>_predictions.npz``, ``<prefix>_metrics.json``, and
    ``<prefix>_per_patient.csv`` into the model's latest run directory,
    where ``<prefix>`` is the split name for SDDB and the dataset name
    for MIT-BIH / INCART. Returns a flat dict of headline numbers used
    by the comparison aggregator.
    """
    run_dir = _latest_run_dir(model_name)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found")

    cfg = _load_merged_config(model_name, override_path=None)
    stride_seconds = float(cfg["stride_seconds"])
    prefix = _artifact_prefix(dataset, split)

    ds = _build_dataset(dataset, split)
    logger.info(
        "[%s/%s] %d windows over %d patients",
        model_name, prefix, len(ds), len(np.unique(ds.patient_ids)),
    )

    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MODELS[model_name]().to(device)
    state = _load_checkpoint(ckpt_path, device, logger)
    model.load_state_dict(_model_state_dict(state))
    model.eval()

    logits, labels = _run_inference(model, loader, device)
    probs = _sigmoid_from_logits(logits)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    tf = compute_threshold_free_metrics(labels, probs)
    auroc_ci = bootstrap_ci(labels, probs, _auroc_metric, n_resamples=500, seed=seed)
    auprc_ci = bootstrap_ci(labels, probs, _auprc_metric, n_resamples=500, seed=seed)

    fixed = compute_threshold_metrics(labels, probs, threshold=0.5)
    fixed["fpr_per_hour"] = compute_fpr_per_hour(
        fp=fixed["fp"], n_negatives=n_neg, stride_seconds=stride_seconds,
    )

    # Val-tuned threshold is reused from SDDB val for every "test-like"
    # eval — SDDB test, MIT-BIH, INCART. Only SDDB val itself opts out
    # (it's the source, not a target).
    val_tuned: dict[str, Any] | None = None
    if not (dataset == "sddb" and split == "val"):
        val_tuned = _val_tuned_metrics(
            run_dir, labels, probs, n_neg, stride_seconds, logger, model_name,
        )

    # Predictions
    preds_path = run_dir / f"{prefix}_predictions.npz"
    np.savez_compressed(
        preds_path,
        logits=logits.astype(np.float32),
        probs=probs.astype(np.float32),
        labels=labels.astype(np.float32),
        patient_ids=np.asarray(ds.patient_ids),
        record_names=np.asarray(ds.record_names),
    )
    logger.info("wrote %s", preds_path)

    # Metrics JSON
    metrics_obj: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset,
        "split": split,
        "n_windows": int(len(ds)),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "stride_s": stride_seconds,
        "threshold_free": {
            "auroc": tf["auroc"],
            "auroc_ci": auroc_ci,
            "auprc": tf["auprc"],
            "auprc_ci": auprc_ci,
        },
        "threshold_fixed_0p5": fixed,
        "threshold_val_tuned_95spec": val_tuned,
        "git_sha": _read_meta_field(run_dir, "git_sha"),
    }
    metrics_path = run_dir / f"{prefix}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(_json_safe(metrics_obj), f, indent=2)
    logger.info("wrote %s", metrics_path)

    # Per-patient CSV
    patient_table = _build_per_patient_table(ds, probs)
    patient_csv_path = run_dir / f"{prefix}_per_patient.csv"
    patient_table.to_csv(patient_csv_path, index=False)
    logger.info("wrote %s", patient_csv_path)

    return {
        "model": model_name,
        "params": _read_param_count(run_dir, model_name),
        "auroc": tf["auroc"],
        "auroc_ci_lo": auroc_ci["lo"],
        "auroc_ci_hi": auroc_ci["hi"],
        "auprc": tf["auprc"],
        "auprc_ci_lo": auprc_ci["lo"],
        "auprc_ci_hi": auprc_ci["hi"],
        "f1_0p5": fixed["f1"],
        "sens_0p5": fixed["sensitivity"],
        "fpr_per_h_0p5": fixed["fpr_per_hour"],
        "thr_95spec": (val_tuned["threshold"] if val_tuned else float("nan")),
        "sens_at_thr": (val_tuned["sensitivity"] if val_tuned else float("nan")),
        "fpr_per_h_at_thr": (val_tuned["fpr_per_hour"] if val_tuned else float("nan")),
        "status": "OK",
    }


def _val_tuned_metrics(
    run_dir: Path,
    target_labels: np.ndarray,
    target_probs: np.ndarray,
    n_neg: int,
    stride_seconds: float,
    logger: logging.Logger,
    model_name: str,
) -> dict[str, Any] | None:
    """Tune the threshold on SDDB val predictions then apply to target."""
    val_path = run_dir / "val_predictions.npz"
    if not val_path.exists():
        logger.warning(
            "%s missing; skipping val-tuned threshold for %s.", val_path, model_name,
        )
        return None
    val_npz = np.load(val_path)
    thr = find_threshold_at_specificity(
        val_npz["labels"], val_npz["probs"], target_spec=0.95,
    )
    result = compute_threshold_metrics(target_labels, target_probs, threshold=thr)
    result["fpr_per_hour"] = compute_fpr_per_hour(
        fp=result["fp"], n_negatives=n_neg, stride_seconds=stride_seconds,
    )
    return result


# ---------- output formatting ----------

def _json_safe(obj: Any) -> Any:
    """Recursively convert NaN/Inf floats to None for strict-JSON compliance.

    Python's ``json.dump`` happily emits literal ``NaN`` / ``Infinity``
    tokens that the JSON spec doesn't allow; downstream parsers (jq,
    most browser fetchers, many language stdlibs) reject them. Walk the
    tree once before dumping and substitute ``None``.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    return obj


def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt_3dp(x: float) -> str:
    return "—" if _is_nan(x) else f"{x:.3f}"


def _fmt_2e(x: float) -> str:
    return "—" if _is_nan(x) else f"{x:.2e}"


def _fmt_1dp(x: float) -> str:
    return "—" if _is_nan(x) else f"{x:.1f}"


def _fmt_int(x: Any) -> str:
    return "—" if x is None else f"{int(x):,}"


def _comparison_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model", "params", "auroc", "auroc_ci", "auprc", "auprc_ci",
        "f1@0.5", "sens@0.5", "fpr_per_h@0.5",
        "thr@95spec", "sens@thr", "fpr_per_h@thr",
    ]
    align = ["---"] + ["---:" for _ in headers[1:]]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        if r.get("status") != "OK":
            cells = [r["model"], f"FAILED: {r.get('error', '?')}"]
            cells += ["—"] * (len(headers) - 2)
            lines.append("| " + " | ".join(cells) + " |")
            continue
        cells = [
            r["model"],
            _fmt_int(r["params"]),
            _fmt_3dp(r["auroc"]),
            f"[{_fmt_3dp(r['auroc_ci_lo'])}, {_fmt_3dp(r['auroc_ci_hi'])}]",
            _fmt_2e(r["auprc"]),
            f"[{_fmt_2e(r['auprc_ci_lo'])}, {_fmt_2e(r['auprc_ci_hi'])}]",
            _fmt_3dp(r["f1_0p5"]),
            _fmt_3dp(r["sens_0p5"]),
            _fmt_1dp(r["fpr_per_h_0p5"]),
            _fmt_3dp(r["thr_95spec"]),
            _fmt_3dp(r["sens_at_thr"]),
            _fmt_1dp(r["fpr_per_h_at_thr"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_comparison(
    rows: list[dict[str, Any]],
    dataset: str,
    split: str | None,
    logger: logging.Logger,
) -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = _artifact_prefix(dataset, split)
    csv_path = RUNS_ROOT / f"comparison_{suffix}.csv"
    md_path = RUNS_ROOT / f"comparison_{suffix}.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    md_path.write_text(_comparison_markdown(rows))
    logger.info("wrote %s and %s", csv_path, md_path)


def main() -> None:
    args = _parse_args()

    # Dataset / split validation — argparse can't express the conditional.
    if args.dataset == "sddb":
        if args.split is None:
            raise SystemExit("--split is required when --dataset sddb")
    elif args.split is not None:
        raise SystemExit(
            f"--split is not applicable when --dataset {args.dataset}; omit it",
        )

    set_seed(args.seed)
    logger = get_logger("evaluate")
    device = _resolve_device(args.device)
    logger.info(
        "device=%s dataset=%s split=%s seed=%d",
        device, args.dataset, args.split, args.seed,
    )

    if args.all:
        rows: list[dict[str, Any]] = []
        for model_name in sorted(MODELS.keys()):
            try:
                rows.append(
                    _evaluate_single(
                        model_name, args.dataset, args.split, device, logger, args.seed,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — one model's failure must not kill the others
                logger.exception("evaluation of %s failed: %s", model_name, exc)
                rows.append({"model": model_name, "status": "FAILED", "error": str(exc)})
        _write_comparison(rows, args.dataset, args.split, logger)
    else:
        _evaluate_single(
            args.model, args.dataset, args.split, device, logger, args.seed,
        )


if __name__ == "__main__":
    main()
