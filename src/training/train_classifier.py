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

