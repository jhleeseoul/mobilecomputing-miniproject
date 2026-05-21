"""Evaluate trained KWS model and generate confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from kws.config import load_config
from kws.constants import build_labels
from kws.data.dataset import make_dataset
from kws.data.manifest import read_manifest
from kws.utils import dump_json, ensure_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KWS model")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--model", type=str, default="small_cnn", choices=["small_cnn", "ds_cnn"])
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def _load_labels(cfg: Dict, manifest_path: Path) -> List[str]:
    label_map_path = manifest_path.parent / "label_map.json"
    if label_map_path.exists():
        import json

        with label_map_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if "labels" in payload:
                return list(payload["labels"])

    return build_labels(cfg.get("commands", []))


def _save_confusion_matrix_plot(cm: np.ndarray, labels: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    manifest_path = Path(args.manifest or cfg["paths"]["manifest"])
    manifest = read_manifest(manifest_path)

    test_ds = make_dataset(
        manifest,
        cfg,
        split="test",
        augment=False,
        shuffle=False,
        max_items=args.max_items if args.max_items > 0 else None,
    )

    model_path = Path(args.model_path) if args.model_path else Path(cfg["paths"]["models_dir"]) / args.model / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    y_true: List[int] = []
    y_pred: List[int] = []

    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy().astype(int).tolist())
        y_pred.extend(np.argmax(preds, axis=1).astype(int).tolist())

    labels = _load_labels(cfg, manifest_path)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )

    report_dir = ensure_dir(Path(cfg["paths"]["reports_dir"]) / args.model)
    np.savetxt(report_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    _save_confusion_matrix_plot(cm, labels, report_dir / "confusion_matrix.png")

    payload = {
        "model_path": str(model_path),
        "accuracy": float(accuracy),
        "num_test_samples": len(y_true),
        "classification_report": cls_report,
    }
    dump_json(payload, report_dir / "eval_report.json")

    print(f"Evaluation complete. accuracy={accuracy:.4f}")
    print(f"Report dir: {report_dir}")


if __name__ == "__main__":
    main()
