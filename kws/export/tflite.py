"""Export Keras models to float32 and int8 TFLite."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from kws.config import load_config
from kws.data.dataset import representative_feature_generator
from kws.utils import dump_json, ensure_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export KWS model to TFLite")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--model", type=str, default="small_cnn", choices=["small_cnn", "ds_cnn"])
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    return parser.parse_args()


def _float32_export(model: tf.keras.Model, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)


def _int8_export(model: tf.keras.Model, cfg: dict, manifest_path: str | None, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    max_samples = int(cfg.get("quantization", {}).get("representative_samples", 200))

    def representative_dataset() -> object:
        for features in representative_feature_generator(
            cfg,
            manifest_path=manifest_path,
            split="train",
            max_samples=max_samples,
        ):
            yield [features]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    int8_model = converter.convert()
    out_path.write_bytes(int8_model)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    model_path = Path(args.model_path) if args.model_path else Path(cfg["paths"]["models_dir"]) / args.model / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    model_dir = ensure_dir(Path(cfg["paths"]["models_dir"]) / args.model)

    float_path = model_dir / "model_float32.tflite"
    int8_path = model_dir / "model_int8.tflite"

    _float32_export(model, float_path)
    _int8_export(model, cfg, args.manifest or None, int8_path)

    payload = {
        "model": args.model,
        "keras_model": str(model_path),
        "float32_tflite": str(float_path),
        "int8_tflite": str(int8_path),
        "float32_bytes": float_path.stat().st_size,
        "int8_bytes": int8_path.stat().st_size,
        "size_reduction_ratio": 1.0 - (int8_path.stat().st_size / float_path.stat().st_size),
    }
    dump_json(payload, model_dir / "tflite_export_report.json")

    print("TFLite export complete")
    print(f"float32: {float_path}")
    print(f"int8: {int8_path}")


if __name__ == "__main__":
    main()
