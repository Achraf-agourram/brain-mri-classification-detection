import json
from pathlib import Path
from ultralytics import YOLO
from src.data.create_yolo_config import create_yolo_config

ROOT_DIR = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT_DIR / "reports" / "detection" / "yolov8n_brain_mri"
BEST_MODEL_PATH = RUN_DIR / "weights" / "best.pt"
REPORT_DIR = ROOT_DIR / "reports" / "detection"


def evaluate_detector():
    if not BEST_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing trained model: {BEST_MODEL_PATH}. Run src.models.train_detector first."
        )

    data_yaml = create_yolo_config()
    model = YOLO(str(BEST_MODEL_PATH))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=4,
        plots=True,
        project=str(REPORT_DIR),
        name="yolov8n_test_evaluation",
        exist_ok=True,
    )

    class_maps = {
        str(model.names[class_id]): float(value)
        for class_id, value in enumerate(metrics.box.maps)
    }
    report = {
        "model": str(BEST_MODEL_PATH),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": class_maps,
    }

    output_path = REPORT_DIR / "test_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    evaluate_detector()
