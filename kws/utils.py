"""Utility helpers for reproducibility and IO."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
