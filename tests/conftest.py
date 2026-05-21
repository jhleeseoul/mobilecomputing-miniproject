from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.io import wavfile

COMMANDS = ["yes", "no", "up", "down", "left", "right", "stop", "go"]
UNKNOWN_WORDS = ["tree", "bed"]


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    int16_audio = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, int16_audio)


def make_dummy_speech_commands(root: Path, sample_rate: int = 16000) -> Path:
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)

    for word_idx, word in enumerate(COMMANDS + UNKNOWN_WORDS):
        freq = 220 + 20 * word_idx
        waveform = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)

        for clip_idx in range(3):
            path = root / word / f"{word}_{clip_idx}.wav"
            _write_wav(path, waveform)

    noise = np.random.default_rng(0).normal(loc=0.0, scale=0.03, size=sample_rate * 3).astype(np.float32)
    _write_wav(root / "_background_noise_" / "noise.wav", noise)

    validation_entries = []
    testing_entries = []
    for word in COMMANDS + UNKNOWN_WORDS:
        validation_entries.append(f"{word}/{word}_0.wav")
        testing_entries.append(f"{word}/{word}_1.wav")

    (root / "validation_list.txt").write_text("\n".join(validation_entries) + "\n", encoding="utf-8")
    (root / "testing_list.txt").write_text("\n".join(testing_entries) + "\n", encoding="utf-8")

    return root


def write_test_config(config_path: Path, dataset_root: Path, artifacts_root: Path) -> Dict:
    cfg = {
        "seed": 123,
        "sample_rate": 16000,
        "clip_duration_ms": 1000,
        "frame_length": 640,
        "frame_step": 320,
        "fft_length": 1024,
        "num_mel_bins": 40,
        "n_mfcc": 13,
        "feature_type": "mfcc",
        "normalize": True,
        "num_classes": 10,
        "commands": COMMANDS,
        "unknown_ratio": 1.0,
        "silence_ratio": 0.1,
        "augment": {
            "enabled": True,
            "time_shift_ms": 50,
            "noise_prob": 0.5,
            "noise_level_min": 0.0,
            "noise_level_max": 0.1,
            "volume_min": 0.8,
            "volume_max": 1.2,
        },
        "train": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "label_smoothing": 0.0,
        },
        "quantization": {"representative_samples": 8, "target": "int8"},
        "paths": {
            "dataset_root": str(dataset_root),
            "manifest": str(artifacts_root / "data" / "manifest.csv"),
            "splits_dir": str(artifacts_root / "data" / "splits"),
            "models_dir": str(artifacts_root / "models"),
            "reports_dir": str(artifacts_root / "reports"),
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def run_cli(args: Tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, *args], cwd=str(cwd), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result
