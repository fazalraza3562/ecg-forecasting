"""YAML config loading, JSON dumping, and run-directory creation helpers."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return the parsed dict."""
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def dump_json(obj: Any, path: str | Path) -> None:
    """Serialize `obj` to `path` as pretty-printed JSON, creating parents as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def make_run_dir(model_name: str, base: str | Path = "runs") -> Path:
    """Create runs/<model_name>/<YYYYMMDD-HHMMSS>/ and return the absolute path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base) / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()
