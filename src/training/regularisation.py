from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data.preprocess_classification import create_datasets

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs" / "l2_comparison"
IMAGE_SIZE = (224, 224)
EPOCHS = 20
SEED = 42

CONFIGURATIONS = [
    {"name": "baseline_no_l2", "l2": 0.0},
    {"name": "l2_1e-4", "l2": 1e-4}
]


def build_model(num_classes, l2_value):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    regularizer = (
        tf.keras.regularizers.l2(l2_value)
        if l2_value > 0
        else None
    )

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizer
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_comparison(histories, train_key, val_key, ylabel, filename):
    plt.figure(figsize=(10, 6))

    for name, history in histories.items():
        epochs = range(1, len(history[train_key]) + 1)

        plt.plot(
            epochs,
            history[train_key],
            label=f"{name} - training"
        )
        plt.plot(
            epochs,
            history[val_key],
            linestyle="--",
            label=f"{name} - validation"
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Training vs validation {ylabel.lower()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset, test_dataset, class_names = (
        create_datasets()
    )

    histories = {}
    results = []

    for configuration in CONFIGURATIONS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        name = configuration["name"]
        model = build_model(
            num_classes=len(class_names),
            l2_value=configuration["l2"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        histories[name] = history.history
        model.save(OUTPUT_DIR / f"{name}.keras")

        best_epoch = int(np.argmax(history.history["val_accuracy"]))

        results.append({
            "configuration": name,
            "l2": configuration["l2"],
            "best_epoch": best_epoch + 1,
            "best_val_accuracy": history.history["val_accuracy"][best_epoch],
            "val_loss_at_best_accuracy": history.history["val_loss"][best_epoch]
        })

    plot_comparison(
        histories,
        "loss",
        "val_loss",
        "Loss",
        "loss_comparison.png"
    )

    plot_comparison(
        histories,
        "accuracy",
        "val_accuracy",
        "Accuracy",
        "accuracy_comparison.png"
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)

    best_configuration = results_df.loc[
        results_df["best_val_accuracy"].idxmax(),
        "configuration"
    ]

    best_model = tf.keras.models.load_model(
        OUTPUT_DIR / f"{best_configuration}.keras"
    )

    test_metrics = best_model.evaluate(
        test_dataset,
        return_dict=True,
        verbose=0
    )

    summary = {
        "selected_configuration": best_configuration,
        "test_metrics": {
            key: float(value)
            for key, value in test_metrics.items()
        }
    }

    (OUTPUT_DIR / "final_test_results.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print(results_df.to_string(index=False))
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()