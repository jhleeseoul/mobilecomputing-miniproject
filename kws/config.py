"""Config loading and validation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path("configs/default_config.json")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to read YAML config files.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if config_file.suffix.lower() in {".yml", ".yaml"}:
        cfg = _load_yaml(config_file)
    else:
        with config_file.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)

    if overrides:
        cfg = _deep_update(cfg, overrides)

    required_top_level = [
        "sample_rate",
        "clip_duration_ms",
        "frame_length",
        "frame_step",
        "num_classes",
        "augment",
        "train",
        "quantization",
    ]
    missing = [key for key in required_top_level if key not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return cfg
