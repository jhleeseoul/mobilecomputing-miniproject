"""Dataset loading from prepared manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Tuple

import numpy as np
import tensorflow as tf

from kws.audio import (
    load_wav,
    random_noise_mix,
    random_time_shift,
    random_volume_scale,
)
from kws.data.manifest import read_manifest
from kws.features import extract_features_np, extract_features_tensor


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    input_shape: Tuple[int, int, int]


def _clip_samples(cfg: Dict) -> int:
    return int(int(cfg["sample_rate"]) * int(cfg["clip_duration_ms"]) / 1000)


def _load_waveform(entry: Dict, cfg: Dict) -> np.ndarray:
    sample_rate = int(cfg["sample_rate"])
    clip_samples = _clip_samples(cfg)
    source_type = str(entry["source_type"])

    if source_type == "zero":
        return np.zeros(clip_samples, dtype=np.float32)

    path = entry["path"]
    start_sample = int(entry.get("start_sample", 0))
    return load_wav(path, sample_rate, clip_samples, start_sample=start_sample)


def _build_noise_bank(entries: Iterable[Dict], cfg: Dict, max_items: int = 64) -> List[np.ndarray]:
    noise_entries = [e for e in entries if e["source_type"] == "noise"]
    if not noise_entries:
        return []

    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 7)
    if len(noise_entries) <= max_items:
        sampled = noise_entries
    else:
        indices = rng.choice(len(noise_entries), size=max_items, replace=False)
        sampled = [noise_entries[int(i)] for i in indices]

    bank: List[np.ndarray] = []
    for entry in sampled:
        bank.append(_load_waveform(entry, cfg))
    return bank


def _augment_waveform(waveform: np.ndarray, noise_bank: List[np.ndarray], cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    aug_cfg = cfg.get("augment", {})
    if not aug_cfg.get("enabled", True):
        return waveform

    max_shift = int(int(cfg["sample_rate"]) * float(aug_cfg.get("time_shift_ms", 0)) / 1000)
    waveform = random_time_shift(waveform, max_shift, rng)

    waveform = random_volume_scale(
        waveform,
        float(aug_cfg.get("volume_min", 1.0)),
        float(aug_cfg.get("volume_max", 1.0)),
        rng,
    )

    noise_prob = float(aug_cfg.get("noise_prob", 0.0))
    if noise_bank and rng.random() < noise_prob:
        waveform = random_noise_mix(
            waveform,
            noise_bank,
            float(aug_cfg.get("noise_level_min", 0.0)),
            float(aug_cfg.get("noise_level_max", 0.1)),
            rng,
        )

    return waveform


def _entries_for_split(
    entries: List[Dict],
    split: str,
    max_items: int | None = None,
    seed: int = 42,
) -> List[Dict]:
    filtered = [entry for entry in entries if entry["split"] == split]
    if max_items is not None and max_items < len(filtered):
        rng = np.random.default_rng(_split_seed(seed, split) + 101)
        indices = rng.choice(len(filtered), size=max_items, replace=False)
        return [filtered[int(idx)] for idx in indices]
    return filtered


def _waveform_generator(
    entries: List[Dict],
    cfg: Dict,
    augment: bool,
    shuffle: bool,
    seed: int,
    noise_bank: List[np.ndarray],
) -> Generator[Tuple[np.ndarray, np.int32], None, None]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(entries))

    if shuffle:
        rng.shuffle(order)

    for idx in order:
        entry = entries[int(idx)]
        waveform = _load_waveform(entry, cfg)

        if augment and entry["label"] != "silence":
            waveform = _augment_waveform(waveform, noise_bank, cfg, rng)

        label = np.int32(entry["label_index"])
        yield waveform.astype(np.float32), label


def _split_seed(base_seed: int, split: str) -> int:
    offsets = {"train": 11, "val": 29, "test": 53}
    return base_seed + offsets.get(split, 97)


def _data_cfg(cfg: Dict) -> Dict:
    return dict(cfg.get("data", {}))


def _use_feature_cache(cfg: Dict, split: str, augment: bool) -> bool:
    data_cfg = _data_cfg(cfg)
    if not bool(data_cfg.get("use_feature_cache", False)):
        return False

    # Train-time augmentation normally changes every epoch.
    # When enabled, caching stores one deterministic augmented view for speed.
    if split == "train" and augment:
        return bool(data_cfg.get("cache_train_augmented_features", True))
    return True


def _shuffle_buffer_size(cfg: Dict, num_examples: int) -> int:
    data_cfg = _data_cfg(cfg)
    requested = int(data_cfg.get("shuffle_buffer_size", 10000))
    return max(1, min(requested, num_examples))


def _feature_cache_paths(
    cfg: Dict,
    split: str,
    max_items: int | None,
    augment: bool,
) -> Tuple[Path, Path, Path]:
    manifest_path = Path(cfg["paths"]["manifest"])
    manifest_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else 0.0

    cache_payload = {
        "split": split,
        "max_items": max_items if max_items is not None else "all",
        "augment": int(augment),
        "seed": int(cfg.get("seed", 42)),
        "sample_rate": int(cfg["sample_rate"]),
        "clip_duration_ms": int(cfg["clip_duration_ms"]),
        "frame_length": int(cfg["frame_length"]),
        "frame_step": int(cfg["frame_step"]),
        "fft_length": int(cfg.get("fft_length", 1024)),
        "feature_type": cfg.get("feature_type", "mfcc"),
        "num_mel_bins": int(cfg.get("num_mel_bins", 40)),
        "n_mfcc": int(cfg.get("n_mfcc", 13)),
        "normalize": bool(cfg.get("normalize", True)),
        "manifest_mtime": manifest_mtime,
    }
    digest = hashlib.sha1(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    cache_dir = Path(_data_cfg(cfg).get("feature_cache_dir", "artifacts/data/feature_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{split}_{digest}"
    return (
        cache_dir / f"{prefix}_x.npy",
        cache_dir / f"{prefix}_y.npy",
        cache_dir / f"{prefix}_meta.json",
    )


def _build_feature_arrays(
    all_entries: List[Dict],
    split_entries: List[Dict],
    cfg: Dict,
    split: str,
    augment: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    noise_bank = _build_noise_bank(all_entries, cfg)
    rng = np.random.default_rng(_split_seed(seed, split))

    features_list: List[np.ndarray] = []
    labels_list: List[np.int32] = []
    for idx, entry in enumerate(split_entries):
        waveform = _load_waveform(entry, cfg)
        if augment and entry["label"] != "silence":
            waveform = _augment_waveform(waveform, noise_bank, cfg, rng)

        features = extract_features_np(waveform, cfg)
        features_list.append(np.expand_dims(features, axis=-1).astype(np.float32))
        labels_list.append(np.int32(entry["label_index"]))

        if (idx + 1) % 5000 == 0:
            print(f"[feature-cache] processed {idx + 1}/{len(split_entries)} for split={split}")

    return np.stack(features_list, axis=0), np.array(labels_list, dtype=np.int32)


def _load_or_build_feature_cache(
    all_entries: List[Dict],
    split_entries: List[Dict],
    cfg: Dict,
    split: str,
    augment: bool,
    seed: int,
    max_items: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    x_path, y_path, meta_path = _feature_cache_paths(cfg, split=split, max_items=max_items, augment=augment)

    if x_path.exists() and y_path.exists():
        features = np.load(x_path, mmap_mode="r")
        labels = np.load(y_path, mmap_mode="r")
        return features, labels

    print(f"[feature-cache] building cache for split={split}, augment={augment}, items={len(split_entries)}")
    features, labels = _build_feature_arrays(all_entries, split_entries, cfg, split=split, augment=augment, seed=seed)
    np.save(x_path, features)
    np.save(y_path, labels)

    meta_payload = {
        "split": split,
        "augment": bool(augment),
        "num_examples": int(labels.shape[0]),
        "feature_shape": list(features.shape[1:]),
        "x_path": str(x_path),
        "y_path": str(y_path),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return features, labels


def make_dataset(
    entries: List[Dict],
    cfg: Dict,
    split: str,
    augment: bool,
    shuffle: bool,
    max_items: int | None,
) -> tf.data.Dataset:
    seed = int(cfg.get("seed", 42))
    split_entries = _entries_for_split(entries, split, max_items=max_items, seed=seed)
    if not split_entries:
        raise ValueError(f"No entries for split={split}. Did you run `python -m kws.data.prepare`?")

    batch_size = int(cfg["train"]["batch_size"])

    if _use_feature_cache(cfg, split=split, augment=augment):
        features, labels = _load_or_build_feature_cache(
            entries,
            split_entries,
            cfg,
            split=split,
            augment=augment,
            seed=seed,
            max_items=max_items,
        )
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            ds = ds.shuffle(
                _shuffle_buffer_size(cfg, int(labels.shape[0])),
                seed=_split_seed(seed, split),
                reshuffle_each_iteration=True,
            )
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    noise_bank = _build_noise_bank(entries, cfg)
    sample_waveform = _load_waveform(split_entries[0], cfg)
    sample_features = extract_features_np(sample_waveform, cfg)
    frame_dim, feat_dim = sample_features.shape

    output_signature = (
        tf.TensorSpec(shape=(sample_waveform.shape[0],), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: _waveform_generator(split_entries, cfg, augment, shuffle, _split_seed(seed, split), noise_bank),
        output_signature=output_signature,
    )
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(split_entries)))

    ds = ds.batch(batch_size)

    def _batch_waveforms_to_features(batch_waveforms: tf.Tensor, batch_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_features = extract_features_tensor(batch_waveforms, cfg)
        batch_features = tf.expand_dims(batch_features, axis=-1)
        return batch_features, batch_labels

    ds = ds.map(_batch_waveforms_to_features, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_dataset_bundle(cfg: Dict, manifest_path: str | Path | None = None, max_items: int | None = None) -> DatasetBundle:
    manifest = read_manifest(manifest_path or cfg["paths"]["manifest"])
    train_ds = make_dataset(manifest, cfg, split="train", augment=True, shuffle=True, max_items=max_items)
    val_ds = make_dataset(manifest, cfg, split="val", augment=False, shuffle=False, max_items=max_items)
    test_ds = make_dataset(manifest, cfg, split="test", augment=False, shuffle=False, max_items=max_items)

    first_batch = next(iter(train_ds.take(1)))
    input_shape = tuple(first_batch[0].shape[1:])  # (frames, bins, 1)

    return DatasetBundle(train=train_ds, val=val_ds, test=test_ds, input_shape=input_shape)


def representative_feature_generator(
    cfg: Dict,
    manifest_path: str | Path | None = None,
    split: str = "train",
    max_samples: int = 200,
) -> Generator[np.ndarray, None, None]:
    entries = read_manifest(manifest_path or cfg["paths"]["manifest"])
    seed = int(cfg.get("seed", 42))
    split_entries = _entries_for_split(entries, split, seed=seed)
    if not split_entries:
        return

    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 99)
    indices = np.arange(len(split_entries))
    rng.shuffle(indices)

    for idx in indices[:max_samples]:
        waveform = _load_waveform(split_entries[int(idx)], cfg)
        features = extract_features_np(waveform, cfg)
        features = np.expand_dims(features, axis=(0, -1)).astype(np.float32)
        yield features
