"""PyTorch Dataset and DataLoader factories backed by the cached window .npz files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ECGWindowDataset(Dataset):
    """In-memory dataset over a single preprocessed ``windows.npz`` file.

    Each item is ``(window, label)`` where ``window`` is shape ``(C, T)``
    and ``label`` is a scalar float tensor in ``{0.0, 1.0}``.
    """

    def __init__(self, npz_path: Path, indices: np.ndarray | None = None) -> None:
        npz = np.load(npz_path, allow_pickle=False)
        signals = npz["signals"]
        labels = npz["labels"]
        patient_ids = npz["patient_ids"]
        record_names = npz["record_names"]

        if indices is not None:
            signals = signals[indices]
            labels = labels[indices]
            patient_ids = patient_ids[indices]
            record_names = record_names[indices]

        # Keep as float32 on CPU; the DataLoader can pin and move to GPU per batch.
        self._signals = np.ascontiguousarray(signals, dtype=np.float32)
        self._labels = np.ascontiguousarray(labels, dtype=np.float32)
        self.patient_ids = patient_ids
        self.record_names = record_names

    @property
    def labels(self) -> np.ndarray:
        """Float32 labels for the rows held by this dataset (used to compute pos_weight)."""
        return self._labels

    def __len__(self) -> int:
        return self._signals.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.from_numpy(self._signals[idx])
        label = torch.tensor(self._labels[idx], dtype=torch.float32)
        return window, label


def _indices_for_patients(patient_ids: np.ndarray, keep: set[str]) -> np.ndarray:
    """Return the row indices whose patient_id is in ``keep``."""
    mask = np.fromiter((pid in keep for pid in patient_ids), dtype=bool, count=len(patient_ids))
    return np.where(mask)[0]


def make_dataloaders(
    processed_root: Path,
    split_json: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, val, test) loaders over the SDDB windows for a patient split."""
    with split_json.open("r") as f:
        split = json.load(f)

    npz_path = processed_root / "sddb" / "windows.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found; run scripts/02_preprocess.sh first")

    # Read patient ids once to compute index lists per split without loading
    # the full signals array three times.
    npz = np.load(npz_path, allow_pickle=False)
    patient_ids = npz["patient_ids"]

    train_idx = _indices_for_patients(patient_ids, set(split["train"]))
    val_idx = _indices_for_patients(patient_ids, set(split["val"]))
    test_idx = _indices_for_patients(patient_ids, set(split["test"]))

    train_ds = ECGWindowDataset(npz_path, indices=train_idx)
    val_ds = ECGWindowDataset(npz_path, indices=val_idx)
    test_ds = ECGWindowDataset(npz_path, indices=test_idx)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader
