"""Manifest read/write helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

MANIFEST_FIELDS = ["path", "label", "label_index", "split", "source_type", "start_sample"]


def write_manifest(entries: Iterable[Dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in entries:
            row = {key: entry.get(key, "") for key in MANIFEST_FIELDS}
            writer.writerow(row)


def read_manifest(path: str | Path) -> List[Dict]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    for row in rows:
        row["label_index"] = int(row["label_index"])
        row["start_sample"] = int(row["start_sample"])

    return rows
