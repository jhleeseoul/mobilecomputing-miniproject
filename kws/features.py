"""Feature extraction utilities (log-mel / MFCC)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf


def _log_mel_tensor(waveform: tf.Tensor, cfg: Dict) -> tf.Tensor:
    sample_rate = int(cfg["sample_rate"])
    frame_length = int(cfg["frame_length"])
    frame_step = int(cfg["frame_step"])
    fft_length = int(cfg.get("fft_length", 1024))
    num_mel_bins = int(cfg.get("num_mel_bins", 40))

    stft = tf.signal.stft(
        signals=waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        pad_end=False,
    )
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=20.0,
        upper_edge_hertz=sample_rate / 2,
    )
    mel_spectrogram = tf.matmul(tf.square(spectrogram), mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram


def extract_features_tensor(waveform: tf.Tensor, cfg: Dict) -> tf.Tensor:
    feature_type = cfg.get("feature_type", "mfcc").lower()
    log_mel = _log_mel_tensor(waveform, cfg)

    if feature_type == "log_mel":
        features = log_mel
    else:
        n_mfcc = int(cfg.get("n_mfcc", 13))
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)
        features = mfcc[..., :n_mfcc]

    if bool(cfg.get("normalize", True)):
        # Keep per-example normalization behavior for both single [T, F] and batched [B, T, F] inputs.
        mean = tf.reduce_mean(features, axis=(-2, -1), keepdims=True)
        std = tf.math.reduce_std(features, axis=(-2, -1), keepdims=True)
        features = (features - mean) / (std + 1e-6)

    return features


def extract_features_np(waveform: np.ndarray, cfg: Dict) -> np.ndarray:
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    features = extract_features_tensor(waveform_tensor, cfg)
    return features.numpy().astype(np.float32)
