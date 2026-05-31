from __future__ import annotations

import json
from pathlib import Path

from kws.class_weights import load_labels, resolve_class_weight


def test_load_labels_prefers_label_map(tmp_path: Path) -> None:
    manifest = tmp_path / "data" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("path,label,label_index,split,source_type,start_sample\n", encoding="utf-8")
    (manifest.parent / "label_map.json").write_text(
        json.dumps({"labels": ["yes", "no", "unknown", "silence"]}),
        encoding="utf-8",
    )

    cfg = {
        "commands": ["up", "down"],
        "paths": {"manifest": str(manifest)},
    }
    assert load_labels(cfg) == ["yes", "no", "unknown", "silence"]


def test_resolve_class_weight_uses_defaults_for_missing_labels() -> None:
    cfg = {
        "train": {
            "class_weights": {
                "down": 1.25,
                "go": 1.15,
            }
        }
    }
    labels = ["yes", "down", "go", "unknown", "silence"]
    out = resolve_class_weight(cfg, labels)
    assert out == {0: 1.0, 1: 1.25, 2: 1.15, 3: 1.0, 4: 1.0}
