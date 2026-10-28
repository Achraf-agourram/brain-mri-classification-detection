import json
from pathlib import Path

import numpy as np
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASSIFICATION_MODELS = {
    "MobileNetV2": {
        "key": "mobilenetv2",
        "path": ROOT_DIR / "models" / "mobilenet" / "best_model.keras",
        "alternate_path": (
            ROOT_DIR / "models" / "mobilenetv2" / "best_model.keras"
        ),
    },
    "ResNet50": {
        "key": "resnet50",
        "path": ROOT_DIR / "models" / "resnet50" / "best_model.keras",
    },
}
DETECTION_MODEL_PATH = ROOT_DIR / "models" / "detection" / "best.pt"


@st.cache_resource
def load_classification_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_detection_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def classify_image(image, model_name: str):
    import tensorflow as tf

    model_info = CLASSIFICATION_MODELS[model_name]
    model_paths = [model_info["path"]]
    if "alternate_path" in model_info:
        model_paths.append(model_info["alternate_path"])
    model_path = next(
        (path for path in model_paths if path.is_file()),
        None,
    )
    if model_path is None:
        raise FileNotFoundError(
            f"Missing {model_name} weights. Expected one of: "
            + ", ".join(str(path) for path in model_paths)
            + ". "
            "Train or provide this model before using classification."
        )

    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    image_array = tf.image.resize(image_array, (224, 224)).numpy()
    if model_info["key"] == "mobilenetv2":
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )
    else:
        image_array = tf.keras.applications.resnet50.preprocess_input(
            image_array
        )

    model = load_classification_model(str(model_path))
    predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            f"{model_name} returned {len(predictions)} scores for "
            f"{len(CLASS_NAMES)} known classes."
        )

    return sorted(
        zip(CLASS_NAMES, (float(score) for score in predictions)),
        key=lambda item: item[1],
        reverse=True,
    )


def detect_objects(image, confidence_threshold: float):
    if not DETECTION_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing detector weights at {DETECTION_MODEL_PATH}. "
            "Train or provide the detection model before using detection."
        )

    model = load_detection_model(str(DETECTION_MODEL_PATH))
    result = model.predict(
        source=image.convert("RGB"),
        conf=confidence_threshold,
        verbose=False,
    )[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        detections.append({
            "class": result.names[class_id],
            "confidence": float(box.conf[0].item()),
            "box": [round(float(value), 1) for value in box.xyxy[0].tolist()],
        })

    annotated_image = result.plot()[:, :, ::-1]
    return annotated_image, detections


def read_classifier_reports():
    reports = []
    report_root = ROOT_DIR / "reports"

    for model_name, model_info in CLASSIFICATION_MODELS.items():
        for metrics_path in sorted(
            report_root.glob(f"config*/{model_info['key']}/metrics.json")
        ):
            with metrics_path.open("r", encoding="utf-8") as file:
                metrics = json.load(file)
            reports.append({
                "configuration": metrics_path.parents[1].name,
                "model": model_name,
                "validation_accuracy": metrics.get(
                    "selected_validation_accuracy"
                ),
                "test_accuracy": metrics.get("test_metrics", {}).get(
                    "accuracy"
                ),
                "test_loss": metrics.get("test_metrics", {}).get("loss"),
            })

    return reports


def read_detection_report():
    report_path = ROOT_DIR / "reports" / "detection" / "test_metrics.json"
    if not report_path.is_file():
        return None

    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file)
