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
L2_REGULARIZATION = 1e-4
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
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(L2_REGULARIZATION)
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


def train_model(model_name):
    model_dir = MODEL_DIR / model_name
    report_dir = REPORT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = create_datasets()

    model, base_model, preprocess_input = build_model(
        model_name,
        len(class_names)
    )

    train_dataset = prepare_dataset(
        train_dataset,
        preprocess_input
    )
    validation_dataset = prepare_dataset(
        validation_dataset,
        preprocess_input
    )
    test_dataset = prepare_dataset(
        test_dataset,
        preprocess_input
    )

    head_checkpoint = model_dir / "head_best.keras"
    fine_tune_checkpoint = model_dir / "fine_tuned_best.keras"
    final_model_path = model_dir / "best_model.keras"

    compile_model(model, INITIAL_LEARNING_RATE)

    head_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=HEAD_EPOCHS,
        callbacks=create_callbacks(head_checkpoint)
    )

    model.save(head_checkpoint)

    pd.DataFrame(head_history.history).to_csv(
        report_dir / "head_history.csv",
        index=False
    )

    head_metrics = get_best_validation_metrics(head_history)

    base_model.trainable = True

    for layer in base_model.layers[:-UNFREEZE_LAST_LAYERS]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, FINE_TUNE_LEARNING_RATE)

    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=create_callbacks(fine_tune_checkpoint)
    )

    model.save(fine_tune_checkpoint)

    pd.DataFrame(fine_tune_history.history).to_csv(
        report_dir / "fine_tuning_history.csv",
        index=False
    )

    fine_tune_metrics = get_best_validation_metrics(
        fine_tune_history
    )

    if fine_tune_metrics["loss"] < head_metrics["loss"]:
        selected_checkpoint = fine_tune_checkpoint
        selected_phase = "fine_tuning"
        selected_validation_metrics = fine_tune_metrics
    else:
        selected_checkpoint = head_checkpoint
        selected_phase = "head_only"
        selected_validation_metrics = head_metrics

    best_model = tf.keras.models.load_model(selected_checkpoint)
    best_model.save(final_model_path)

    test_metrics = best_model.evaluate(
        test_dataset,
        return_dict=True,
        verbose=1
    )

    results = {
        "model": model_name,
        "class_names": list(class_names),
        "selected_phase": selected_phase,
        "head_validation_loss": head_metrics["loss"],
        "head_validation_accuracy": head_metrics["accuracy"],
        "fine_tuning_validation_loss": fine_tune_metrics["loss"],
        "fine_tuning_validation_accuracy": fine_tune_metrics[
            "accuracy"
        ],
        "selected_validation_loss": selected_validation_metrics[
            "loss"
        ],
        "selected_validation_accuracy": selected_validation_metrics[
            "accuracy"
        ],
        "test_metrics": {
            key: float(value)
            for key, value in test_metrics.items()
        },
        "model_path": str(final_model_path)
    }

    with open(report_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    with open(report_dir / "classes.json", "w", encoding="utf-8") as file:
        json.dump(list(class_names), file, indent=4)

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mobilenetv2", "resnet50"],
        required=True
    )
    arguments = parser.parse_args()

    train_model(arguments.model)
