from __future__ import annotations

import csv
import json
from pathlib import Path

from tests.conftest import COMMANDS, make_dummy_speech_commands, run_cli, write_test_config


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prepare_generates_fixed_label_mapping_and_splits(tmp_path: Path) -> None:
    dataset_root = make_dummy_speech_commands(tmp_path / "speech_commands")
    cfg_path = tmp_path / "config.json"
    artifacts_root = tmp_path / "artifacts"
    write_test_config(cfg_path, dataset_root, artifacts_root)

    run_cli(("-m", "kws.data.prepare", "--config", str(cfg_path), "--dataset-root", str(dataset_root)), cwd=Path.cwd())

    label_map_path = artifacts_root / "data" / "label_map.json"
    assert label_map_path.exists()

    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    expected_labels = COMMANDS + ["unknown", "silence"]
    assert payload["labels"] == expected_labels

    manifest_path = artifacts_root / "data" / "manifest.csv"
    rows = _read_csv(manifest_path)
    splits = {row["split"] for row in rows}
    assert splits == {"train", "val", "test"}

    assert any(row["label"] == "unknown" for row in rows)
    assert any(row["label"] == "silence" for row in rows)


def test_prepare_is_reproducible_with_fixed_seed(tmp_path: Path) -> None:
    dataset_root = make_dummy_speech_commands(tmp_path / "speech_commands")
    cfg_path = tmp_path / "config.json"
    artifacts_root = tmp_path / "artifacts"
    write_test_config(cfg_path, dataset_root, artifacts_root)

    run_cli(("-m", "kws.data.prepare", "--config", str(cfg_path), "--dataset-root", str(dataset_root)), cwd=Path.cwd())
    first_manifest = (artifacts_root / "data" / "manifest.csv").read_text(encoding="utf-8")

    run_cli(("-m", "kws.data.prepare", "--config", str(cfg_path), "--dataset-root", str(dataset_root)), cwd=Path.cwd())
    second_manifest = (artifacts_root / "data" / "manifest.csv").read_text(encoding="utf-8")

    assert first_manifest == second_manifest
