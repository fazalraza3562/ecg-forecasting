"""Download SDDB, MIT-BIH, and INCART recordings from PhysioNet into data/raw/."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger("download")

# PhysioNet database identifiers we care about.
DATASETS: tuple[str, ...] = ("sddb", "mitdb", "incartdb")


def _missing_records(db_name: str, target_dir: Path) -> list[str]:
    """Return record names from `db_name` whose .hea is not yet present in `target_dir`."""
    import wfdb  # imported lazily so --help works without the dependency installed

    all_records = wfdb.io.get_record_list(db_name)
    return [r for r in all_records if not (target_dir / f"{r}.hea").exists()]


def download_dataset(db_name: str, data_root: Path) -> None:
    """Fetch any records of `db_name` that are not already on disk."""
    import wfdb

    target_dir = data_root / db_name
    target_dir.mkdir(parents=True, exist_ok=True)

    missing = _missing_records(db_name, target_dir)
    if not missing:
        LOGGER.info("%s: all records already present in %s; nothing to download.",
                    db_name, target_dir)
        return

    LOGGER.info("%s: downloading %d record(s) into %s", db_name, len(missing), target_dir)
    wfdb.io.dl_database(db_name, str(target_dir), records=missing)
    LOGGER.info("%s: finished.", db_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SDDB, MIT-BIH, and INCART recordings from PhysioNet."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to the shared YAML config (used to resolve --data-root default).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        metavar="NAME",
        help=f"Datasets to fetch. Default: all of {', '.join(DATASETS)}.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the data root directory. Default: read `data_root` from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_root = args.data_root if args.data_root is not None else Path(cfg["data_root"])

    LOGGER.info("Data root: %s", data_root.resolve())
    for db_name in args.datasets:
        download_dataset(db_name, data_root)


if __name__ == "__main__":
    main()
