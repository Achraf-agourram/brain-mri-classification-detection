from pathlib import Path
import pandas as pd
import tensorflow as tf
from src.data.prepare_dataset import get_records, create_splits, CLASS_NAMES

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "mri_classification"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_RATIO = 0.10
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    ),
    tf.keras.layers.RandomContrast(0.10)
])
