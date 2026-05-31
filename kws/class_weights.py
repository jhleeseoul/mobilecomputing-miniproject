"""Helpers for resolving class-weight config into Keras class_weight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from kws.constants import build_labels


def load_labels(cfg: Dict, manifest_path: str | Path | None = None) -> List[str]:
    manifest = Path(manifest_path or cfg["paths"]["manifest"])
    label_map_path = manifest.parent / "label_map.json"
    if label_map_path.exists():
        with label_map_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        labels = payload.get("labels")
        if isinstance(labels, list) and labels:
            return [str(item) for item in labels]
    return build_labels(cfg.get("commands", []))


def resolve_class_weight(cfg: Dict, labels: List[str]) -> Dict[int, float] | None:
    train_cfg = cfg.get("train", {})
    raw = train_cfg.get("class_weights", {})
    if not isinstance(raw, dict) or not raw:
        return None

    out: Dict[int, float] = {}
    for idx, label in enumerate(labels):
        weight = float(raw.get(label, 1.0))
        if weight <= 0.0:
            raise ValueError(f"class weight for label '{label}' must be >0, got {weight}")
        out[idx] = weight
    return out
