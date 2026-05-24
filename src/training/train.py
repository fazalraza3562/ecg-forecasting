"""Model-agnostic trainer for the six ECG forecasting models.

CLI:

    python -m src.training.train --model <name> [--config <path>] [--seed <int>] [--device <cpu|cuda|auto>]

The trainer holds no model-specific knowledge. It picks the architecture
from the ``--model`` flag via a name -> class registry, reads every
hyperparameter from the merged config (``config/default.yaml`` overlaid
with ``config/<model>.yaml``), and writes everything it produces under
``runs/<model>/<timestamp>/``: ``best.pt`` checkpoint, ``config.yaml``
snapshot, ``train.log``, ``meta.json``, and on Ctrl+C an ``interrupted.pt``.

The training loss is focal (gamma from the config). The early-stopping
metric is validation AUPRC, which is the metric that actually matters at
0.7 % positive prevalence — AUROC is too easy to inflate at this
imbalance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import make_dataloaders
from src.models.baseline_lstm import BaselineLSTM
from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.models.inception1d import InceptionTime1D
from src.models.resnet1d import ResNet1D
from src.models.tcn import TCN1D
from src.models.transformer import TransformerEncoderModel
from src.training.losses import FocalLoss, compute_pos_weight_from_labels
from src.training.scheduler import make_cosine_schedule_with_warmup
from src.utils.io import dump_json, load_yaml, make_run_dir
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


# Name -> class map. Adding a new model means importing it above and
# adding one entry here; nothing in this file should special-case any
# architecture beyond this registry.
MODELS: dict[str, Callable[[], nn.Module]] = {
    "baseline_lstm": BaselineLSTM,
    "cnn_lstm_attention": CNNLSTMAttention,
    # Ablation that shares the CNN-LSTM body with cnn_lstm_attention but
    # swaps the attention head for mean pooling. The lambda is the
    # dispatch — nothing else in the pipeline needs to know it's the
    # same class.
    "cnn_lstm_noattention": lambda: CNNLSTMAttention(use_attention=False),
    "transformer": TransformerEncoderModel,
    "resnet1d": ResNet1D,
    "inception1d": InceptionTime1D,
    "tcn": TCN1D,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
SPLIT_JSON = PROCESSED_ROOT / "sddb" / "split.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model", required=True, choices=sorted(MODELS.keys()),
        help="Which architecture to train.",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to a model config; defaults to config/<model>.yaml.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed field in the merged config.",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="auto picks CUDA if available, otherwise CPU.",
    )
    return p.parse_args()


def _load_merged_config(model_name: str, override_path: str | None) -> dict[str, Any]:
    """Load default.yaml then overlay the model-specific config on top."""
    cfg = load_yaml(DEFAULT_CONFIG_PATH)
    model_cfg_path = (
        Path(override_path)
        if override_path is not None
        else REPO_ROOT / "config" / f"{model_name}.yaml"
    )
    if model_cfg_path.exists():
        overrides = load_yaml(model_cfg_path) or {}
        cfg.update(overrides)
    return cfg


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return sha or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auprc: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_auprc": val_auprc,
        },
        path,
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_seen = 0

    use_amp = scaler is not None
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(windows)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(windows)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n_seen += bs

    return running_loss / max(n_seen, 1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return (mean_loss, AUROC, AUPRC) on the loader.

    Returns NaN for AUROC/AUPRC if the loader contains only one class —
    sklearn would otherwise raise. That happens on small validation
    splits early in development; reporting NaN is more honest than
    silently fabricating a metric.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    running_loss = 0.0
    n_seen = 0

    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        logits = model(windows)
        loss = loss_fn(logits, labels)
        bs = labels.size(0)
        running_loss += loss.item() * bs
        n_seen += bs
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    y_logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    mean_loss = running_loss / max(n_seen, 1)

    if len(np.unique(y_true)) < 2:
        return mean_loss, float("nan"), float("nan")

    y_score = 1.0 / (1.0 + np.exp(-y_logits))  # sigmoid for the metric inputs
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    return mean_loss, auroc, auprc


def main() -> None:
    args = _parse_args()
    cfg = _load_merged_config(args.model, args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    set_seed(seed)

    device = _resolve_device(args.device)
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"

    run_dir = make_run_dir(args.model)
    logger = get_logger("train", log_dir=run_dir)
    logger.info("model=%s seed=%d device=%s amp=%s", args.model, seed, device, use_amp)
    logger.info("run_dir=%s", run_dir)

    # Snapshot the *effective* config (post-merge, post-CLI overrides) so
    # the run is reproducible without referring back to the source configs.
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump({**cfg, "seed": seed}, f, sort_keys=True)

    train_loader, val_loader, _ = make_dataloaders(
        processed_root=PROCESSED_ROOT,
        split_json=SPLIT_JSON,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
    )
    logger.info(
        "windows: train=%d val=%d",
        len(train_loader.dataset), len(val_loader.dataset),
    )

    train_labels = train_loader.dataset.labels
    pos_weight = compute_pos_weight_from_labels(train_labels)
    n_pos = int((train_labels > 0).sum())
    logger.info(
        "train positives: %d / %d (%.3f%%) -> pos_weight=%.2f (informational; focal loss in use)",
        n_pos, len(train_labels), 100.0 * n_pos / max(len(train_labels), 1), pos_weight,
    )

    model = MODELS[args.model]().to(device)
    logger.info("model params: %d", _count_params(model))

    loss_fn = FocalLoss(gamma=float(cfg["focal_gamma"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    total_steps = max(len(train_loader) * int(cfg["epochs"]), 1)
    # Warmup duration is configurable via cfg["warmup_epochs"]; the legacy
    # 10%-of-total-steps default kicks in if the key is absent so older
    # configs that don't set it keep their previous behavior.
    warmup_epochs = cfg.get("warmup_epochs")
    if warmup_epochs is not None:
        warmup_steps = max(int(float(warmup_epochs) * len(train_loader)), 1)
    else:
        warmup_steps = max(int(0.1 * total_steps), 1)
        warmup_epochs = warmup_steps / max(len(train_loader), 1)
    logger.info(
        "warmup: %s epochs = %d steps", warmup_epochs, warmup_steps,
    )
    scheduler = make_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_auprc = -float("inf")
    best_epoch = -1
    patience = 0
    epochs_trained = 0
    t0 = time.time()
    interrupted = False

    try:
        for epoch in range(1, int(cfg["epochs"]) + 1):
            train_loss = _train_one_epoch(
                model, train_loader, loss_fn,
                optimizer, scheduler, scaler, device,
            )
            val_loss, val_auroc, val_auprc = _evaluate(
                model, val_loader, loss_fn, device,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "epoch %d: train_loss=%.5f val_loss=%.5f val_auroc=%.4f val_auprc=%.4f lr=%.2e",
                epoch, train_loss, val_loss, val_auroc, val_auprc, current_lr,
            )
            epochs_trained = epoch

            # Compare with NaN safety: if val_auprc is NaN (single-class
            # val split) we don't update best and we don't tick patience.
            if not np.isnan(val_auprc) and val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                patience = 0
                _save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, val_auprc)
                logger.info("  new best AUPRC -> saved best.pt")
            else:
                patience += 1
                if patience >= int(cfg["early_stopping_patience"]):
                    logger.info(
                        "early stopping at epoch %d (patience=%d, best AUPRC=%.4f @ epoch %d)",
                        epoch, patience, best_auprc, best_epoch,
                    )
                    break
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("interrupted at epoch %d; saving interrupted.pt", epochs_trained)
        _save_checkpoint(
            run_dir / "interrupted.pt", model, optimizer,
            epochs_trained, float(best_auprc) if best_auprc > -float("inf") else float("nan"),
        )
    finally:
        meta = {
            "model": args.model,
            "config": {**cfg, "seed": seed},
            "seed": seed,
            "git_sha": _git_sha(),
            "best_val_auprc": (
                float(best_auprc) if best_auprc > -float("inf") else None
            ),
            "best_epoch": best_epoch if best_epoch > 0 else None,
            "epochs_trained": epochs_trained,
            "training_seconds": time.time() - t0,
            "interrupted": interrupted,
        }
        dump_json(meta, run_dir / "meta.json")
        logger.info("wrote meta.json (training_seconds=%.1f)", meta["training_seconds"])

    if interrupted:
        # Re-raise so the shell sees the SIGINT exit code, matching the
        # convention that Ctrl+C produces a non-zero exit.
        raise KeyboardInterrupt


if __name__ == "__main__":
    main()
