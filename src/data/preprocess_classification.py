from pathlib import Path
import pandas as pd
import tensorflow as tf
from src.data.prepare_dataset import get_records, create_splits, CLASS_NAMES

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_RATIO = 0.10
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

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


def load_and_resize(image_path, class_id):
    image_data = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, class_id


def augment_and_normalize(image, class_id):
    image = augmentation(image, training=True)
    image = image / 255.0
    return image, class_id


def normalize(image, class_id):
    image = image / 255.0
    return image, class_id


def create_dataset(records, training=False):
    dataframe = pd.DataFrame(records)
    image_paths = dataframe["image_path"].astype(str).to_numpy()
    class_ids = dataframe["class_id"].to_numpy(dtype="int32")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, class_ids))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(load_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True)

    if training:
        dataset = dataset.map(augment_and_normalize, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE, deterministic=True)

    options = tf.data.Options()
    options.deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset


def create_datasets():
    records, orphan_labels = get_records()
    splits = create_splits(records)

    class_names = CLASS_NAMES

    train_dataset = create_dataset(
        splits["train"],
        training=True
    )

    validation_dataset = create_dataset(
        splits["validation"]
    )

    test_dataset = create_dataset(
        splits["test"]
    )

    return train_dataset, validation_dataset, test_dataset, class_names