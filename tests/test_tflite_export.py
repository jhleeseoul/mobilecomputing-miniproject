from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from kws.features import extract_features_np
from kws.models.small_cnn import build_small_cnn
from tests.conftest import make_dummy_speech_commands, run_cli, write_test_config


def test_float32_and_int8_tflite_are_loadable(tmp_path: Path) -> None:
    dataset_root = make_dummy_speech_commands(tmp_path / "speech_commands")
    cfg_path = tmp_path / "config.json"
    artifacts_root = tmp_path / "artifacts"
    cfg = write_test_config(cfg_path, dataset_root, artifacts_root)

    run_cli(("-m", "kws.data.prepare", "--config", str(cfg_path), "--dataset-root", str(dataset_root)), cwd=Path.cwd())

    clip_samples = int(cfg["sample_rate"] * cfg["clip_duration_ms"] / 1000)
    dummy_waveform = np.zeros(clip_samples, dtype=np.float32)
    dummy_features = extract_features_np(dummy_waveform, cfg)
    input_shape = (dummy_features.shape[0], dummy_features.shape[1], 1)

    model = build_small_cnn(input_shape=input_shape, num_classes=cfg["num_classes"])
    model_path = artifacts_root / "models" / "small_cnn" / "best_model.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    run_cli(
        (
            "-m",
            "kws.export.tflite",
            "--config",
            str(cfg_path),
            "--model",
            "small_cnn",
            "--model-path",
            str(model_path),
            "--manifest",
            str(artifacts_root / "data" / "manifest.csv"),
        ),
        cwd=Path.cwd(),
    )

    float_path = artifacts_root / "models" / "small_cnn" / "model_float32.tflite"
    int8_path = artifacts_root / "models" / "small_cnn" / "model_int8.tflite"

    assert float_path.exists()
    assert int8_path.exists()

    float_interpreter = tf.lite.Interpreter(model_path=str(float_path))
    int8_interpreter = tf.lite.Interpreter(model_path=str(int8_path))

    float_interpreter.allocate_tensors()
    int8_interpreter.allocate_tensors()
