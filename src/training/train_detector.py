import json
from pathlib import Path
from ultralytics import YOLO
from src.data.create_yolo_config import ANNOTATION_CLASSES, create_yolo_config

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

    return model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=10,
        patience=12,
        workers=4,
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
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )


if __name__ == "__main__":
    train_detector()
