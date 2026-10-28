import json
from pathlib import Path
from ultralytics import YOLO
from src.data.create_yolo_config import ANNOTATION_CLASSES, create_yolo_config
import shutil

ROOT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT_DIR / "reports" / "detection"


def train_detector():
    data_yaml = create_yolo_config()
    quality_path = REPORT_DIR / "dataset_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    train_box_counts = quality["accepted"]["train"]["boxes_by_class"]
    missing_classes = [
        class_name for class_name in ANNOTATION_CLASSES.values()
        if train_box_counts.get(class_name, 0) == 0
    ]

    if missing_classes:
        raise ValueError(
            "The training split has no valid bounding boxes for: "
            + ", ".join(missing_classes)
            + ". Review data/processed/mri_detection/rejected_images.csv and fix the folder/annotation mapping before training."
        )

    model = YOLO("yolov8n.pt")

    results =  model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        pretrained=True,
        cache=False,
        seed=42,
        deterministic=True,
        project=str(REPORT_DIR),
        name="yolov8n_brain_mri",
        exist_ok=True,
        plots=True,
        save=True,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        device=0
    )

    best_model_path = REPORT_DIR / "yolov8n_brain_mri" / "weights" / "best.pt"
    models_dir = ROOT_DIR / "models" / "detection"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model_path, models_dir / "best.pt")

    return results


if __name__ == "__main__":
    train_detector()
