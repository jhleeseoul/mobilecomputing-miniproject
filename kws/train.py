"""Train KWS models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import tensorflow as tf

from kws.config import load_config
from kws.data.dataset import build_dataset_bundle
from kws.models.factory import build_model
from kws.utils import dump_json, ensure_dir, set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KWS model")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--model", type=str, default="small_cnn", choices=["small_cnn", "ds_cnn"])
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs from config if >0")
    parser.add_argument("--max-items", type=int, default=0, help="Limit items per split for smoke testing")
    return parser.parse_args()


def _to_float_dict(metrics: Dict) -> Dict:
    out: Dict = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            out[key] = [float(v) for v in value]
        else:
            out[key] = float(value)
    return out


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    max_items = args.max_items if args.max_items > 0 else None
    bundle = build_dataset_bundle(cfg, manifest_path=args.manifest or None, max_items=max_items)
    train_cardinality = int(tf.data.experimental.cardinality(bundle.train).numpy())
    configured_spe = int(cfg["train"].get("steps_per_execution", 1))
    if train_cardinality > 0:
        steps_per_execution = max(1, min(configured_spe, train_cardinality))
    else:
        steps_per_execution = max(1, configured_spe)

    model = build_model(args.model, bundle.input_shape, int(cfg["num_classes"]))
    learning_rate = float(cfg["train"]["learning_rate"])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        steps_per_execution=steps_per_execution,
    )

    epochs = args.epochs if args.epochs > 0 else int(cfg["train"]["epochs"])

    model_dir = ensure_dir(Path(cfg["paths"]["models_dir"]) / args.model)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=int(cfg["train"].get("early_stopping_patience", 5)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(bundle.train, validation_data=bundle.val, epochs=epochs, callbacks=callbacks)

    test_loss, test_acc = model.evaluate(bundle.test, verbose=0)
    model.save(model_dir / "last_model.keras")

    history_payload = _to_float_dict(history.history)
    dump_json(history_payload, model_dir / "history.json")

    train_summary = {
        "model": args.model,
        "input_shape": list(bundle.input_shape),
        "train_steps_per_epoch": train_cardinality,
        "steps_per_execution": steps_per_execution,
        "epochs_requested": epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }
    dump_json(train_summary, model_dir / "train_summary.json")

    summary_path = model_dir / "model_summary.txt"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        model.summary(print_fn=lambda line: summary_file.write(line + "\n"))

    print(f"Training complete. Best model: {model_dir / 'best_model.keras'}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
