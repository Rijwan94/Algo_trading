from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_json(path: str | Path, default: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
