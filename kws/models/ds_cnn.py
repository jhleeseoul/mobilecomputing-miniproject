"""DS-CNN style compact model with depthwise separable blocks."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def _ds_block(x: tf.Tensor, filters: int, stride: Tuple[int, int] = (1, 1)) -> tf.Tensor:
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_ds_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="features")

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = _ds_block(x, 64)
    x = _ds_block(x, 64)
    x = _ds_block(x, 96, stride=(2, 1))
    x = _ds_block(x, 96)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ds_cnn")
