"""Assert that train, validation, and test splits contain disjoint patient IDs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "data" / "processed" / "sddb" / "split.json"
SDDB_RAW = REPO_ROOT / "data" / "raw" / "sddb"


def _annotated_sddb_patient_ids() -> set[str]:
    """SDDB patients that survive preprocessing — those with BOTH .hea and .atr files.

    SDDB ships 23 records, but only 12 of them have an accompanying ``.atr``
    annotation file (records 30-32, 34-36, 41, 45, 46, 49, 51, 52 at the time
    of writing). The other 11 records (33, 37, 38, 39, 40, 42, 43, 44, 47, 48,
    50) have signal data but no ground-truth onset labels, so
    ``preprocess.preprocess_record`` skips them and they correctly never make
    it into the split. The "expected" set this test compares the split against
    must therefore be the *annotatable* patients, not every patient on disk —
    otherwise the unannotated records show up as spurious "missing" failures.
    """
    return {
        f"sddb/{hea.stem}"
        for hea in SDDB_RAW.glob("*.hea")
        if hea.with_suffix(".atr").exists()
    }


def test_split_disjoint_and_complete() -> None:
    if not SPLIT_PATH.exists():
        pytest.skip("preprocessing not yet run")
    if not SDDB_RAW.exists():
        pytest.skip("raw SDDB not downloaded")

    with SPLIT_PATH.open("r") as f:
        split = json.load(f)

    train = set(split["train"])
    val = set(split["val"])
    test = set(split["test"])

    assert train.isdisjoint(val), f"train/val overlap: {sorted(train & val)}"
    assert train.isdisjoint(test), f"train/test overlap: {sorted(train & test)}"
    assert val.isdisjoint(test), f"val/test overlap: {sorted(val & test)}"

    union = train | val | test
    expected = _annotated_sddb_patient_ids()
    # Compare against the annotatable subset (see helper docstring). The split
    # must cover exactly that subset: no on-disk patient with valid annotations
    # may be missing, and the split must not invent patients that aren't there.
    missing = expected - union
    extra = union - expected
    assert not extra, f"split references non-existent patients: {sorted(extra)}"
    assert not missing, f"annotated SDDB patients missing from split: {sorted(missing)}"
