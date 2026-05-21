"""Prepare Speech Commands manifest with known/unknown/silence mapping."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.io import wavfile

from kws.config import load_config
from kws.constants import build_labels
from kws.data.manifest import write_manifest
from kws.utils import dump_json, ensure_dir, set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Speech Commands manifest.")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--download", action="store_true", help="Download dataset via kagglehub if root missing.")
    return parser.parse_args()


def _find_dataset_root(path: Path) -> Path:
    candidates = []
    for candidate in [path, *path.rglob("*")]:
        if not candidate.is_dir():
            continue
        if (candidate / "validation_list.txt").exists() and (candidate / "testing_list.txt").exists():
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            "Could not find Speech Commands root with validation_list.txt/testing_list.txt"
        )

    candidates.sort(key=lambda p: len(p.as_posix()))
    return candidates[0]


def _resolve_dataset_root(config_root: str, arg_root: str, allow_download: bool) -> Path:
    if arg_root:
        return _find_dataset_root(Path(arg_root).expanduser().resolve())

    if config_root:
        return _find_dataset_root(Path(config_root).expanduser().resolve())

    if not allow_download:
        raise ValueError(
            "Dataset root not provided. Use --dataset-root, set paths.dataset_root in config, or pass --download"
        )

    import kagglehub  # type: ignore

    download_path = Path(kagglehub.dataset_download("neehakurelli/google-speech-commands"))
    return _find_dataset_root(download_path.resolve())


def _read_split_file(dataset_root: Path, filename: str) -> set[str]:
    split_path = dataset_root / filename
    with split_path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _get_split(rel_path: str, validation_set: set[str], test_set: set[str]) -> str:
    if rel_path in validation_set:
        return "val"
    if rel_path in test_set:
        return "test"
    return "train"


def _noise_lengths(noise_files: Sequence[Path], sample_rate: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for wav_path in noise_files:
        sr, samples = wavfile.read(str(wav_path))
        if sr != sample_rate:
            # We only need an approximate length in target SR for offset sampling.
            ratio = float(sample_rate) / float(sr)
            out[str(wav_path)] = int(len(samples) * ratio)
        else:
            out[str(wav_path)] = int(len(samples))
    return out


def _build_entry(path: Path | None, label: str, label_index: int, split: str, source_type: str, start_sample: int) -> Dict:
    return {
        "path": str(path.resolve()) if path else "",
        "label": label,
        "label_index": label_index,
        "split": split,
        "source_type": source_type,
        "start_sample": int(start_sample),
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    dataset_root = _resolve_dataset_root(cfg["paths"].get("dataset_root", ""), args.dataset_root, args.download)
    commands = list(cfg.get("commands") or [])
    labels = build_labels(commands)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    sample_rate = int(cfg["sample_rate"])
    clip_samples = int(sample_rate * cfg["clip_duration_ms"] / 1000)
    unknown_ratio = float(cfg.get("unknown_ratio", 1.0))
    silence_ratio = float(cfg.get("silence_ratio", 0.1))

    validation_set = _read_split_file(dataset_root, "validation_list.txt")
    test_set = _read_split_file(dataset_root, "testing_list.txt")

    background_dir = dataset_root / "_background_noise_"
    noise_files = sorted(background_dir.glob("*.wav")) if background_dir.exists() else []
    noise_length_map = _noise_lengths(noise_files, sample_rate) if noise_files else {}

    known_entries_by_split: Dict[str, List[Dict]] = defaultdict(list)
    unknown_pool_by_split: Dict[str, List[Path]] = defaultdict(list)

    for wav_path in dataset_root.rglob("*.wav"):
        if "_background_noise_" in wav_path.parts:
            continue
        rel = wav_path.relative_to(dataset_root).as_posix()
        label = wav_path.parent.name
        split = _get_split(rel, validation_set, test_set)

        if label in commands:
            known_entries_by_split[split].append(
                _build_entry(wav_path, label, label_to_idx[label], split, "clip", 0)
            )
        else:
            unknown_pool_by_split[split].append(wav_path)

    manifest_entries: List[Dict] = []
    for split in ("train", "val", "test"):
        known_entries = known_entries_by_split[split]
        manifest_entries.extend(known_entries)

        unknown_target_count = int(len(known_entries) * unknown_ratio)
        unknown_pool = unknown_pool_by_split[split]
        if unknown_pool and unknown_target_count > 0:
            sample_count = min(len(unknown_pool), unknown_target_count)
            selected_idx = rng.choice(len(unknown_pool), size=sample_count, replace=False)
            for idx in selected_idx:
                path = unknown_pool[int(idx)]
                manifest_entries.append(
                    _build_entry(path, "unknown", label_to_idx["unknown"], split, "clip", 0)
                )

        current_count = len([e for e in manifest_entries if e["split"] == split])
        silence_count = int(current_count * silence_ratio)
        for _ in range(silence_count):
            if noise_files:
                noise_path = noise_files[int(rng.integers(0, len(noise_files)))]
                max_start = max(noise_length_map[str(noise_path)] - clip_samples, 0)
                start_sample = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                manifest_entries.append(
                    _build_entry(noise_path, "silence", label_to_idx["silence"], split, "noise", start_sample)
                )
            else:
                manifest_entries.append(
                    _build_entry(None, "silence", label_to_idx["silence"], split, "zero", 0)
                )

    manifest_path = Path(cfg["paths"]["manifest"])
    splits_dir = ensure_dir(cfg["paths"]["splits_dir"])

    write_manifest(manifest_entries, manifest_path)
    for split in ("train", "val", "test"):
        split_entries = [entry for entry in manifest_entries if entry["split"] == split]
        write_manifest(split_entries, splits_dir / f"{split}.csv")

    dump_json({"labels": labels, "label_to_index": label_to_idx}, manifest_path.parent / "label_map.json")
    dump_json(
        {
            "dataset_root": str(dataset_root),
            "sample_rate": sample_rate,
            "clip_samples": clip_samples,
            "split_counts": Counter(entry["split"] for entry in manifest_entries),
            "label_counts": Counter(entry["label"] for entry in manifest_entries),
        },
        manifest_path.parent / "manifest_meta.json",
    )

    split_counts = Counter(entry["split"] for entry in manifest_entries)
    label_counts = Counter(entry["label"] for entry in manifest_entries)

    print("Prepared manifest:", manifest_path)
    print("Dataset root:", dataset_root)
    print("Split counts:", dict(split_counts))
    print("Label counts:", dict(label_counts))


if __name__ == "__main__":
    main()
