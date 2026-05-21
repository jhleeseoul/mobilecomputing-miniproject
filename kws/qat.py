"""Quantization-Aware Training (QAT) pipeline for small_cnn."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from kws.benchmark import _materialize_test_set, _run_tflite_inference
from kws.config import load_config
from kws.data.dataset import build_dataset_bundle, representative_feature_generator
from kws.utils import dump_json, ensure_dir, set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QAT fine-tuning and export int8 TFLite")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--base-model", type=str, default="small_cnn")
    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument("--qat-model-name", type=str, default="small_cnn_qat")
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--qat-learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--representative-samples", type=int, default=400)
    return parser.parse_args()


def _build_small_cnn_tf_keras(input_shape: Tuple[int, int, int], num_classes: int):
    import tf_keras as keras

    inputs = keras.Input(shape=input_shape, name="features")

    x = keras.layers.Conv2D(24, (3, 3), padding="same", activation="relu", name="conv2d")(inputs)
    x = keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="max_pooling2d")(x)

    x = keras.layers.Conv2D(48, (3, 3), padding="same", activation="relu", name="conv2d_1")(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)

    x = keras.layers.Conv2D(96, (3, 3), padding="same", activation="relu", name="conv2d_2")(x)
    x = keras.layers.BatchNormalization(name="batch_normalization_2")(x)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)

    x = keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="small_cnn")


def _qat_ready_model_from_base(base_model: tf.keras.Model, cfg: Dict):
    import tf_keras as keras
    import tensorflow_model_optimization as tfmot

    if base_model.name != "small_cnn":
        raise ValueError(
            "QAT helper currently supports only small_cnn architecture. "
            f"Loaded model name: {base_model.name}"
        )

    legacy_model = _build_small_cnn_tf_keras(
        input_shape=tuple(base_model.input_shape[1:]),
        num_classes=int(cfg["num_classes"]),
    )
    legacy_model.set_weights(base_model.get_weights())

    def annotate(layer):
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense)):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated_model = keras.models.clone_model(legacy_model, clone_function=annotate)
    with tfmot.quantization.keras.quantize_scope():
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    return qat_model


def _export_qat_int8_tflite(
    qat_model,
    cfg: Dict,
    out_path: Path,
    manifest_path: str | None,
    representative_samples: int,
) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for features in representative_feature_generator(
            cfg,
            manifest_path=manifest_path,
            split="train",
            max_samples=representative_samples,
        ):
            yield [features]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)


def _ensure_float32_tflite(base_model: tf.keras.Model, float_path: Path) -> None:
    if float_path.exists():
        return
    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    float_path.write_bytes(converter.convert())


def _benchmark_pair(cfg: Dict, float_path: Path, int8_path: Path, manifest_path: Path, max_items: int | None) -> Dict:
    x_test, y_test = _materialize_test_set(cfg, manifest_path, max_items=max_items)

    float_interpreter = tf.lite.Interpreter(model_path=str(float_path))
    int8_interpreter = tf.lite.Interpreter(model_path=str(int8_path))

    float_logits, float_lat = _run_tflite_inference(float_interpreter, x_test)
    int8_logits, int8_lat = _run_tflite_inference(int8_interpreter, x_test)

    y_pred_float = np.argmax(float_logits, axis=1)
    y_pred_int8 = np.argmax(int8_logits, axis=1)

    report = {
        "num_test_samples": int(len(y_test)),
        "float32": {
            "accuracy": float(accuracy_score(y_test, y_pred_float)),
            "avg_latency_ms": float(float_lat.mean()),
            "p95_latency_ms": float(np.percentile(float_lat, 95)),
            "size_kb": float(float_path.stat().st_size / 1024.0),
        },
        "int8_qat": {
            "accuracy": float(accuracy_score(y_test, y_pred_int8)),
            "avg_latency_ms": float(int8_lat.mean()),
            "p95_latency_ms": float(np.percentile(int8_lat, 95)),
            "size_kb": float(int8_path.stat().st_size / 1024.0),
        },
    }
    report["delta"] = {
        "accuracy_drop_pp": (report["float32"]["accuracy"] - report["int8_qat"]["accuracy"]) * 100.0,
        "size_reduction_ratio": 1.0 - (report["int8_qat"]["size_kb"] / report["float32"]["size_kb"]),
        "latency_delta_ms": report["int8_qat"]["avg_latency_ms"] - report["float32"]["avg_latency_ms"],
    }
    return report


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    max_items = args.max_items if args.max_items > 0 else None
    manifest_path = args.manifest or cfg["paths"]["manifest"]

    base_model_path = (
        Path(args.base_model_path)
        if args.base_model_path
        else Path(cfg["paths"]["models_dir"]) / args.base_model / "best_model.keras"
    )
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    # Load Keras model before importing tfmot to avoid keras/tf_keras loader conflicts.
    base_model = tf.keras.models.load_model(base_model_path)

    bundle = build_dataset_bundle(cfg, manifest_path=manifest_path, max_items=max_items)
    train_cardinality = int(tf.data.experimental.cardinality(bundle.train).numpy())

    qat_model = _qat_ready_model_from_base(base_model, cfg)

    import tf_keras as keras

    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(args.qat_learning_rate)),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    qat_model_dir = ensure_dir(Path(cfg["paths"]["models_dir"]) / args.qat_model_name)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=str(qat_model_dir / "best_model_qat.weights.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    history = qat_model.fit(
        bundle.train,
        validation_data=bundle.val,
        epochs=int(args.qat_epochs),
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = qat_model.evaluate(bundle.test, verbose=0)

    float_path = Path(cfg["paths"]["models_dir"]) / args.base_model / "model_float32.tflite"
    _ensure_float32_tflite(base_model, float_path)

    int8_qat_path = qat_model_dir / "model_int8_qat.tflite"
    _export_qat_int8_tflite(
        qat_model,
        cfg,
        out_path=int8_qat_path,
        manifest_path=manifest_path,
        representative_samples=int(args.representative_samples),
    )

    benchmark = _benchmark_pair(
        cfg,
        float_path=float_path,
        int8_path=int8_qat_path,
        manifest_path=Path(manifest_path),
        max_items=max_items,
    )

    summary = {
        "base_model_path": str(base_model_path),
        "qat_model_name": args.qat_model_name,
        "train_steps_per_epoch": train_cardinality,
        "qat_epochs_requested": int(args.qat_epochs),
        "qat_epochs_ran": len(history.history.get("loss", [])),
        "qat_learning_rate": float(args.qat_learning_rate),
        "qat_test_loss": float(test_loss),
        "qat_test_accuracy": float(test_acc),
        "float32_tflite_path": str(float_path),
        "int8_qat_tflite_path": str(int8_qat_path),
        "benchmark": benchmark,
    }

    dump_json(summary, qat_model_dir / "qat_summary.json")

    report_dir = ensure_dir(Path(cfg["paths"]["reports_dir"]) / args.qat_model_name)
    dump_json(summary, report_dir / "qat_report.json")

    print("QAT complete")
    print(f"QAT model dir: {qat_model_dir}")
    print(f"QAT test accuracy: {test_acc:.4f}")
    print(
        "QAT int8 vs float32: "
        f"float_acc={benchmark['float32']['accuracy']:.4f}, "
        f"int8_qat_acc={benchmark['int8_qat']['accuracy']:.4f}, "
        f"drop_pp={benchmark['delta']['accuracy_drop_pp']:.4f}"
    )


if __name__ == "__main__":
    main()
