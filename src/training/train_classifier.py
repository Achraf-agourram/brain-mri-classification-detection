import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.data.preprocess_classification import (
    AUTOTUNE,
    IMAGE_SIZE,
    create_datasets
)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models" / "transfer_learning"
REPORT_DIR = ROOT_DIR / "reports" / "transfer_learning"

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 8
UNFREEZE_LAST_LAYERS = 30
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
SEED = 42

tf.keras.utils.set_random_seed(SEED)


def load_backbone(model_name):
    input_shape = (*IMAGE_SIZE, 3)

    if model_name == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        preprocess_input = (
            tf.keras.applications.mobilenet_v2.preprocess_input
        )

    elif model_name == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        preprocess_input = (
            tf.keras.applications.resnet50.preprocess_input
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return base_model, preprocess_input


def prepare_dataset(dataset, preprocess_input):
    return dataset.map(
        lambda images, labels: (
            preprocess_input(images * 255.0),
            labels
        ),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)


def build_model(model_name, num_classes):
    base_model, preprocess_input = load_backbone(model_name)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = base_model(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.3)(features)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(features)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model, preprocess_input


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def create_callbacks(checkpoint_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7
        )
    ]


def get_best_validation_metrics(history):
    best_index = min(
        range(len(history.history["val_loss"])),
        key=lambda index: history.history["val_loss"][index]
    )

    return {
        "loss": float(history.history["val_loss"][best_index]),
        "accuracy": float(
            history.history["val_accuracy"][best_index]
        )
    }

