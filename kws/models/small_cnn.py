"""Small CNN baseline model."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def build_small_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="features")

    x = tf.keras.layers.Conv2D(24, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(48, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="small_cnn")
