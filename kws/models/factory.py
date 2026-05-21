"""Model factory."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from kws.models.ds_cnn import build_ds_cnn
from kws.models.small_cnn import build_small_cnn


def build_model(model_name: str, input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    model_name = model_name.lower().strip()
    if model_name == "small_cnn":
        return build_small_cnn(input_shape, num_classes)
    if model_name == "ds_cnn":
        return build_ds_cnn(input_shape, num_classes)

    raise ValueError(f"Unsupported model type: {model_name}")
