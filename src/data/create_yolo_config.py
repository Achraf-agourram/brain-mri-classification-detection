from collections import Counter
import csv
import json
import math
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT_DIR / "data" / "processed" / "mri_dataset"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "mri_detection"
REPORT_DIR = ROOT_DIR / "reports" / "detection"
ANNOTATION_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary",
}
CLASS_TO_ID = {name: class_id for class_id, name in ANNOTATION_CLASSES.items()}
NO_TUMOR_CLASS = "No Tumor"
TOLERANCE = 1e-5


def parse_yolo_label(label_path):
    objects = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.split()

            if not values:
                continue

            if len(values) != 5:
                raise ValueError(f"line {line_number}: expected five values")

            try:
                class_value = float(values[0])
                coordinates = [float(value) for value in values[1:]]
            except ValueError as error:
                raise ValueError(f"line {line_number}: non-numeric value") from error

            if not math.isfinite(class_value) or not class_value.is_integer():
                raise ValueError(f"line {line_number}: invalid class ID")

            class_id = int(class_value)

            if class_id not in ANNOTATION_CLASSES:
                raise ValueError(f"line {line_number}: unknown detection class ID {class_id}")

            x_center, y_center, width, height = coordinates

            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(f"line {line_number}: non-finite coordinate")

            if not (
                -TOLERANCE <= x_center <= 1 + TOLERANCE
                and -TOLERANCE <= y_center <= 1 + TOLERANCE
                and 0 < width <= 1 + TOLERANCE
                and 0 < height <= 1 + TOLERANCE
            ):
                raise ValueError(f"line {line_number}: coordinates are not normalized")

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            if (
                x1 < -TOLERANCE
                or y1 < -TOLERANCE
                or x2 > 1 + TOLERANCE
                or y2 > 1 + TOLERANCE
            ):
                raise ValueError(f"line {line_number}: bounding box exceeds image bounds")

            objects.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            })

    return objects


def validate_record(row):
    image_path = Path(row["image_path"])
    folder_class = row["class_name"].strip()
    raw_label_path = row.get("label_path", "").strip()
    label_path = Path(raw_label_path) if raw_label_path else None

    if not image_path.is_file():
        return None, "missing_image"

    if label_path is None or not label_path.is_file():
        if folder_class == NO_TUMOR_CLASS:
            return {
                "image_path": str(image_path.resolve()),
                "label_path": "",
                "folder_class": folder_class,
                "detection_class": "background",
                "object_count": 0,
                "annotation_status": "negative_background_image",
            }, None
        return None, "missing_tumor_annotation"

    try:
        objects = parse_yolo_label(label_path)
    except ValueError as error:
        return None, f"invalid_label: {error}"

    if not objects:
        if folder_class == NO_TUMOR_CLASS:
            return {
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "folder_class": folder_class,
                "detection_class": "background",
                "object_count": 0,
                "annotation_status": "negative_background_image",
            }, None
        return None, "empty_annotation_for_tumor_image"

    if folder_class == NO_TUMOR_CLASS:
        return None, "no_tumor_image_has_bounding_boxes_review_required"

    if folder_class not in CLASS_TO_ID:
        return None, "unknown_folder_class"

    class_ids = {item["class_id"] for item in objects}

    if len(class_ids) > 1:
        return None, "multiple_detection_classes_in_one_image"

    detection_class_id = next(iter(class_ids))
    expected_class_id = CLASS_TO_ID[folder_class]

    if detection_class_id != expected_class_id:
        return None, "folder_annotation_class_mismatch"

    return {
        "image_path": str(image_path.resolve()),
        "label_path": str(label_path.resolve()),
        "folder_class": folder_class,
        "detection_class": ANNOTATION_CLASSES[detection_class_id],
        "object_count": len(objects),
        "annotation_status": "valid_tumor_boxes",
    }, None


def create_yolo_config():
    split_mapping = {
        "train": "train.csv",
        "val": "validation.csv",
        "test": "test.csv",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    accepted_summary = {}
    rejected_counts = Counter()
    rejected_rows = []

    for split_name, csv_name in split_mapping.items():
        source_csv = SPLIT_DIR / csv_name

        if not source_csv.is_file():
            raise FileNotFoundError(
                f"Missing split manifest: {source_csv}. Run src.data.prepare_dataset first."
            )

        accepted_rows = []

        with source_csv.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                accepted, reason = validate_record(row)

                if accepted is None:
                    rejected_counts[reason.split(":", 1)[0]] += 1
                    rejected_rows.append({
                        "split": split_name,
                        "image_path": row.get("image_path", ""),
                        "folder_class": row.get("class_name", ""),
                        "reason": reason,
                    })
                    continue

                accepted["split"] = split_name
                accepted_rows.append(accepted)

        if not accepted_rows:
            raise ValueError(f"No valid images remain in the {split_name} split")

        image_list_path = OUTPUT_DIR / f"{split_name}.txt"
        image_list_path.write_text(
            "\n".join(row["image_path"] for row in accepted_rows) + "\n",
            encoding="utf-8",
        )

        with (OUTPUT_DIR / f"{split_name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=[
                "image_path",
                "label_path",
                "folder_class",
                "detection_class",
                "object_count",
                "annotation_status",
                "split",
            ])
            writer.writeheader()
            writer.writerows(accepted_rows)

        accepted_summary[split_name] = {
            "total_images": len(accepted_rows),
            "images_with_tumor_boxes": sum(
                row["object_count"] > 0 for row in accepted_rows
            ),
            "negative_background_images": sum(
                row["object_count"] == 0 for row in accepted_rows
            ),
            "bounding_boxes": sum(row["object_count"] for row in accepted_rows),
            "boxes_by_class": {
                class_name: sum(
                    row["object_count"]
                    for row in accepted_rows
                    if row["detection_class"] == class_name
                )
                for class_name in ANNOTATION_CLASSES.values()
            },
        }

    with (OUTPUT_DIR / "rejected_images.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "image_path", "folder_class", "reason"],
        )
        writer.writeheader()
        writer.writerows(rejected_rows)

    config = {
        "path": str(ROOT_DIR),
        "train": str((OUTPUT_DIR / "train.txt").relative_to(ROOT_DIR)),
        "val": str((OUTPUT_DIR / "val.txt").relative_to(ROOT_DIR)),
        "test": str((OUTPUT_DIR / "test.txt").relative_to(ROOT_DIR)),
        "names": ANNOTATION_CLASSES,
    }

    config_path = OUTPUT_DIR / "yolo_data.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    summary_path = REPORT_DIR / "dataset_quality.json"
    summary_path.write_text(
        json.dumps({
            "classes": ANNOTATION_CLASSES,
            "accepted": accepted_summary,
            "rejected_image_count": len(rejected_rows),
            "rejected_reasons": dict(rejected_counts),
            "negative_image_policy": (
                "No Tumor images without boxes are background examples, not a detection class."
            ),
            "mapping_review_policy": (
                "Images in the No Tumor folder that contain bounding boxes, and all folder/annotation class mismatches, are excluded for review rather than silently reassigned. If the Pituitary class ID was confused with the No Tumor folder index, correct the copied dataset mapping and rerun dataset preparation before training."
            ),
        }, indent=2),
        encoding="utf-8",
    )

    return config_path


if __name__ == "__main__":
    create_yolo_config()
