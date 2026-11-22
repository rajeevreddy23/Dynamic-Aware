"""Configuration loading utilities for EAT-Sched."""

from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    """Return the root path of the project."""
    # This file is in src/eatsched/utils/config.py -> 3 levels up to project root
    return Path(__file__).resolve().parent.parent.parent.parent


def load_yaml(filepath: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    path = Path(filepath)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content or {}
