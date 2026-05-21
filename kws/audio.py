"""Audio loading and augmentation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    if samples.dtype == np.int32:
        return samples.astype(np.float32) / 2147483648.0
    if samples.dtype == np.uint8:
        return (samples.astype(np.float32) - 128.0) / 128.0
    return samples.astype(np.float32)


def load_wav(
    wav_path: str | Path,
    target_sample_rate: int,
    target_num_samples: int,
    start_sample: int = 0,
) -> np.ndarray:
    sample_rate, raw = wavfile.read(str(wav_path))
    waveform = _to_float32(raw)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if sample_rate != target_sample_rate:
        waveform = resample_poly(waveform, target_sample_rate, sample_rate).astype(np.float32)

    if start_sample > 0:
        waveform = waveform[start_sample:]

    return fix_length(waveform, target_num_samples)


def fix_length(waveform: np.ndarray, target_num_samples: int) -> np.ndarray:
    if len(waveform) >= target_num_samples:
        return waveform[:target_num_samples].astype(np.float32)

    pad_size = target_num_samples - len(waveform)
    return np.pad(waveform, (0, pad_size), mode="constant").astype(np.float32)


def random_time_shift(waveform: np.ndarray, max_shift_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_shift_samples <= 0:
        return waveform

    shift = int(rng.integers(-max_shift_samples, max_shift_samples + 1))
    if shift == 0:
        return waveform

    out = np.zeros_like(waveform)
    if shift > 0:
        out[shift:] = waveform[:-shift]
    else:
        out[:shift] = waveform[-shift:]
    return out


def random_volume_scale(
    waveform: np.ndarray,
    min_scale: float,
    max_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if min_scale <= 0 or max_scale <= 0:
        return waveform

    scale = float(rng.uniform(min_scale, max_scale))
    return np.clip(waveform * scale, -1.0, 1.0)


def random_noise_mix(
    waveform: np.ndarray,
    noise_bank: Iterable[np.ndarray],
    min_level: float,
    max_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise_bank = list(noise_bank)
    if not noise_bank:
        return waveform

    noise = noise_bank[int(rng.integers(0, len(noise_bank)))]
    if len(noise) != len(waveform):
        noise = fix_length(noise, len(waveform))

    level = float(rng.uniform(min_level, max_level))
    mixed = waveform + level * noise
    return np.clip(mixed, -1.0, 1.0)
