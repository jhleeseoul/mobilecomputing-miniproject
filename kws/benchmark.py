"""Benchmark float32/int8 TFLite models on prepared test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

from kws.config import load_config
from kws.constants import build_labels
from kws.data.manifest import read_manifest
from kws.features import extract_features_np
from kws.audio import load_wav
from kws.utils import dump_json, ensure_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TFLite models")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--model", type=str, default="small_cnn", choices=["small_cnn", "ds_cnn"])
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def _labels(cfg: Dict, manifest_path: Path) -> List[str]:
    label_map = manifest_path.parent / "label_map.json"
    if label_map.exists():
        with label_map.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            labels = payload.get("labels")
            if labels:
                return list(labels)
    return build_labels(cfg.get("commands", []))


def _materialize_test_set(cfg: Dict, manifest_path: Path, max_items: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    entries = [entry for entry in read_manifest(manifest_path) if entry["split"] == "test"]
    if max_items and max_items < len(entries):
        rng = np.random.default_rng(int(cfg.get("seed", 42)) + 211)
        indices = rng.choice(len(entries), size=max_items, replace=False)
        entries = [entries[int(idx)] for idx in indices]

    sample_rate = int(cfg["sample_rate"])
    clip_samples = int(sample_rate * int(cfg["clip_duration_ms"]) / 1000)

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    for entry in entries:
        if entry["source_type"] == "zero":
            waveform = np.zeros(clip_samples, dtype=np.float32)
        else:
            waveform = load_wav(entry["path"], sample_rate, clip_samples, start_sample=int(entry.get("start_sample", 0)))

        features = extract_features_np(waveform, cfg)
        features = np.expand_dims(features, axis=-1).astype(np.float32)
        features_list.append(features)
        labels_list.append(int(entry["label_index"]))

    return np.stack(features_list, axis=0), np.array(labels_list, dtype=np.int32)


def _run_tflite_inference(interpreter: tf.lite.Interpreter, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    logits = []
    latencies_ms = []

    in_scale, in_zero_point = input_details.get("quantization", (0.0, 0))
    out_scale, out_zero_point = output_details.get("quantization", (0.0, 0))

    for sample in inputs:
        sample = np.expand_dims(sample, axis=0)

        if input_details["dtype"] == np.int8:
            if in_scale == 0:
                raise ValueError("Invalid int8 input quantization scale=0")
            quantized = np.round(sample / in_scale + in_zero_point).astype(np.int8)
            input_tensor = quantized
        else:
            input_tensor = sample.astype(input_details["dtype"])

        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], input_tensor)
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        output = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] == np.int8:
            if out_scale == 0:
                raise ValueError("Invalid int8 output quantization scale=0")
            output = (output.astype(np.float32) - out_zero_point) * out_scale
        logits.append(output[0].astype(np.float32))

    return np.stack(logits, axis=0), np.array(latencies_ms, dtype=np.float32)


def _write_markdown_table(report: Dict, path: Path) -> None:
    content = "\n".join(
        [
            "| Model | Accuracy | Avg Latency (ms) | Size (KB) |",
            "|---|---:|---:|---:|",
            f"| float32 | {report['float32']['accuracy']:.4f} | {report['float32']['avg_latency_ms']:.2f} | {report['float32']['size_kb']:.1f} |",
            f"| int8 | {report['int8']['accuracy']:.4f} | {report['int8']['avg_latency_ms']:.2f} | {report['int8']['size_kb']:.1f} |",
            "",
            f"- Accuracy drop (int8 - float32): {report['delta']['accuracy_drop']:.4f}",
            f"- Size reduction ratio: {report['delta']['size_reduction_ratio']:.4f}",
            f"- Latency delta (int8 - float32): {report['delta']['latency_delta_ms']:.2f} ms",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    model_dir = Path(cfg["paths"]["models_dir"]) / args.model
    float_path = model_dir / "model_float32.tflite"
    int8_path = model_dir / "model_int8.tflite"

    if not float_path.exists() or not int8_path.exists():
        raise FileNotFoundError("Missing TFLite models. Run `python -m kws.export.tflite` first.")

    manifest_path = Path(args.manifest or cfg["paths"]["manifest"])
    x_test, y_test = _materialize_test_set(cfg, manifest_path, max_items=args.max_items if args.max_items > 0 else None)

    float_interpreter = tf.lite.Interpreter(model_path=str(float_path))
    int8_interpreter = tf.lite.Interpreter(model_path=str(int8_path))

    float_logits, float_lat = _run_tflite_inference(float_interpreter, x_test)
    int8_logits, int8_lat = _run_tflite_inference(int8_interpreter, x_test)

    y_pred_float = np.argmax(float_logits, axis=1)
    y_pred_int8 = np.argmax(int8_logits, axis=1)

    labels = _labels(cfg, manifest_path)

    report = {
        "model": args.model,
        "num_test_samples": int(len(y_test)),
        "float32": {
            "accuracy": float(accuracy_score(y_test, y_pred_float)),
            "avg_latency_ms": float(float_lat.mean()),
            "p95_latency_ms": float(np.percentile(float_lat, 95)),
            "size_kb": float(float_path.stat().st_size / 1024.0),
            "confusion_matrix": confusion_matrix(y_test, y_pred_float, labels=list(range(len(labels)))).tolist(),
        },
        "int8": {
            "accuracy": float(accuracy_score(y_test, y_pred_int8)),
            "avg_latency_ms": float(int8_lat.mean()),
            "p95_latency_ms": float(np.percentile(int8_lat, 95)),
            "size_kb": float(int8_path.stat().st_size / 1024.0),
            "confusion_matrix": confusion_matrix(y_test, y_pred_int8, labels=list(range(len(labels)))).tolist(),
        },
    }
    report["delta"] = {
        "accuracy_drop": report["int8"]["accuracy"] - report["float32"]["accuracy"],
        "size_reduction_ratio": 1.0 - (report["int8"]["size_kb"] / report["float32"]["size_kb"]),
        "latency_delta_ms": report["int8"]["avg_latency_ms"] - report["float32"]["avg_latency_ms"],
    }

    report_dir = ensure_dir(Path(cfg["paths"]["reports_dir"]) / args.model)
    dump_json(report, report_dir / "benchmark_report.json")
    _write_markdown_table(report, report_dir / "quantization_comparison.md")

    print("Benchmark complete")
    print(f"float32 acc={report['float32']['accuracy']:.4f}, latency={report['float32']['avg_latency_ms']:.2f}ms")
    print(f"int8 acc={report['int8']['accuracy']:.4f}, latency={report['int8']['avg_latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
