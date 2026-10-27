from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT_DIR / "data" / "processed" / "mri_dataset"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "mri_detection"
YOLO_DATASET_DIR = OUTPUT_DIR / "yolo_dataset"
REPORT_DIR = ROOT_DIR / "reports" / "detection"

SOURCE_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}
ANNOTATION_CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary",
}
SOURCE_TO_DETECTION_ID = {
    0: 0,
    1: 1,
    3: 2,
}
CLASS_TO_SOURCE_ID = {
    class_name: class_id
    for class_id, class_name in SOURCE_CLASSES.items()
    if class_name != "No Tumor"
}
NO_TUMOR_CLASS = "No Tumor"
TOLERANCE = 1e-5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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

            source_class_id = int(class_value)

            if source_class_id not in SOURCE_CLASSES:
                raise ValueError(
                    f"line {line_number}: unknown source class ID {source_class_id}"
                )

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

            x1 = min(1.0, max(0.0, x1))
            y1 = min(1.0, max(0.0, y1))
            x2 = min(1.0, max(0.0, x2))
            y2 = min(1.0, max(0.0, y2))

            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"line {line_number}: bounding box has no area")

            objects.append({
                "source_class_id": source_class_id,
                "x_center": (x1 + x2) / 2,
                "y_center": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
            })

    return objects


def validate_record(row):
    image_path = Path(row["image_path"])
    folder_class = row.get("class_name", "").strip()
    raw_label_path = row.get("label_path", "").strip()
    label_path = Path(raw_label_path) if raw_label_path else None

    if not image_path.is_file():
        return None, "missing_image"

    if folder_class not in SOURCE_CLASSES.values():
        return None, "unknown_folder_class"

    if label_path is None or not label_path.is_file():
        if folder_class == NO_TUMOR_CLASS:
            return {
                "image_path": image_path.resolve(),
                "source_label_path": "",
                "folder_class": folder_class,
                "objects": [],
                "detection_classes": [],
                "class_counts": {name: 0 for name in ANNOTATION_CLASSES.values()},
                "ignored_annotation_boxes": 0,
                "annotation_status": "negative_background_image_without_label_file",
            }, None
        return None, "missing_tumor_annotation"

    try:
        objects = parse_yolo_label(label_path)
    except ValueError as error:
        return None, f"invalid_label: {error}"

    if folder_class == NO_TUMOR_CLASS:
        if objects and any(item["source_class_id"] != 2 for item in objects):
            return None, "no_tumor_folder_contains_tumor_class_boxes_review_required"

        return {
            "image_path": image_path.resolve(),
            "source_label_path": str(label_path.resolve()),
            "folder_class": folder_class,
            "objects": [],
            "detection_classes": [],
            "class_counts": {name: 0 for name in ANNOTATION_CLASSES.values()},
            "ignored_annotation_boxes": len(objects),
            "annotation_status": (
                "negative_background_image_no_tumor_boxes_removed"
                if objects
                else "negative_background_image_empty_label"
            ),
        }, None

    if not objects:
        return None, "empty_annotation_for_tumor_image"

    source_class_ids = {item["source_class_id"] for item in objects}

    if 2 in source_class_ids:
        return None, "no_tumor_class_id_inside_tumor_image_review_required"

    expected_source_id = CLASS_TO_SOURCE_ID[folder_class]

    if expected_source_id not in source_class_ids:
        return None, "folder_annotation_class_mismatch"

    class_counts = {name: 0 for name in ANNOTATION_CLASSES.values()}
    normalized_objects = []

    for item in objects:
        detection_class_id = SOURCE_TO_DETECTION_ID[item["source_class_id"]]
        detection_class_name = ANNOTATION_CLASSES[detection_class_id]
        class_counts[detection_class_name] += 1
        normalized_objects.append({
            "class_id": detection_class_id,
            "x_center": item["x_center"],
            "y_center": item["y_center"],
            "width": item["width"],
            "height": item["height"],
        })

    detection_classes = [
        class_name
        for class_name in ANNOTATION_CLASSES.values()
        if class_counts[class_name] > 0
    ]

    return {
        "image_path": image_path.resolve(),
        "source_label_path": str(label_path.resolve()),
        "folder_class": folder_class,
        "objects": normalized_objects,
        "detection_classes": detection_classes,
        "class_counts": class_counts,
        "ignored_annotation_boxes": 0,
        "annotation_status": "valid_tumor_boxes",
    }, None


def create_image_link(source_path, destination_path):
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        try:
            if os.path.samefile(source_path, destination_path):
                return
        except OSError:
            pass
        destination_path.unlink()

    try:
        os.link(source_path, destination_path)
    except OSError as hardlink_error:
        try:
            destination_path.symlink_to(source_path)
        except OSError as symlink_error:
            raise OSError(
                "Could not create a hard link or symbolic link for the image. "
                "The detection dataset uses links to avoid copying raw image files. "
                f"Source: {source_path}; destination: {destination_path}."
            ) from symlink_error


def write_yolo_label(label_path, objects):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{item['class_id']} {item['x_center']:.8f} {item['y_center']:.8f} "
        f"{item['width']:.8f} {item['height']:.8f}"
        for item in objects
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def create_yolo_config():
    split_mapping = {
        "train": "train.csv",
        "val": "validation.csv",
        "test": "test.csv",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if YOLO_DATASET_DIR.exists():
        shutil.rmtree(YOLO_DATASET_DIR)

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

        with source_csv.open("r", newline="", encoding="utf-8-sig") as file:
            for row_number, row in enumerate(csv.DictReader(file), start=2):
                accepted, reason = validate_record(row)

                if accepted is None:
                    rejected_counts[reason] += 1
                    rejected_rows.append({
                        "split": split_name,
                        "row_number": row_number,
                        "image_path": row.get("image_path", ""),
                        "folder_class": row.get("class_name", ""),
                        "reason": reason,
                    })
                    continue

                original_path = accepted["image_path"]
                unique_key = f"{split_name}|{original_path.as_posix()}".casefold()
                image_id = hashlib.sha256(unique_key.encode("utf-8")).hexdigest()[:12]
                image_name = f"{image_id}{original_path.suffix.lower()}"
                linked_image_path = YOLO_DATASET_DIR / split_name / "images" / image_name
                generated_label_path = YOLO_DATASET_DIR / split_name / "labels" / f"{image_id}.txt"

                if original_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    rejected_counts["unsupported_image_extension"] += 1
                    rejected_rows.append({
                        "split": split_name,
                        "row_number": row_number,
                        "image_path": str(original_path),
                        "folder_class": accepted["folder_class"],
                        "reason": "unsupported_image_extension",
                    })
                    continue

                create_image_link(original_path, linked_image_path)
                write_yolo_label(generated_label_path, accepted["objects"])

                class_counts = accepted["class_counts"]
                accepted_rows.append({
                    "image_path": str(linked_image_path.resolve()),
                    "label_path": str(generated_label_path.resolve()),
                    "source_image_path": str(original_path),
                    "source_label_path": accepted["source_label_path"],
                    "folder_class": accepted["folder_class"],
                    "detection_classes": ";".join(accepted["detection_classes"]),
                    "object_count": len(accepted["objects"]),
                    "class_counts": json.dumps(class_counts, sort_keys=True),
                    "ignored_annotation_boxes": accepted["ignored_annotation_boxes"],
                    "annotation_status": accepted["annotation_status"],
                    "split": split_name,
                })

        if not accepted_rows:
            raise ValueError(f"No valid images remain in the {split_name} split")

        image_list_path = OUTPUT_DIR / f"{split_name}.txt"
        image_list_path.write_text(
            "\n".join(row["image_path"] for row in accepted_rows) + "\n",
            encoding="utf-8",
        )

        write_csv(
            OUTPUT_DIR / f"{split_name}.csv",
            accepted_rows,
            [
                "image_path",
                "label_path",
                "source_image_path",
                "source_label_path",
                "folder_class",
                "detection_classes",
                "object_count",
                "class_counts",
                "ignored_annotation_boxes",
                "annotation_status",
                "split",
            ],
        )

        box_counts = {name: 0 for name in ANNOTATION_CLASSES.values()}
        for row in accepted_rows:
            counts = json.loads(row["class_counts"])
            for class_name in box_counts:
                box_counts[class_name] += counts[class_name]

        accepted_summary[split_name] = {
            "total_images": len(accepted_rows),
            "images_with_tumor_boxes": sum(row["object_count"] > 0 for row in accepted_rows),
            "negative_background_images": sum(row["object_count"] == 0 for row in accepted_rows),
            "bounding_boxes": sum(row["object_count"] for row in accepted_rows),
            "boxes_by_class": box_counts,
            "ignored_no_tumor_boxes": sum(row["ignored_annotation_boxes"] for row in accepted_rows),
            "annotation_status_counts": dict(Counter(row["annotation_status"] for row in accepted_rows)),
        }

    write_csv(
        OUTPUT_DIR / "rejected_images.csv",
        rejected_rows,
        ["split", "row_number", "image_path", "folder_class", "reason"],
    )

    config = {
        "path": str(YOLO_DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": ANNOTATION_CLASSES,
    }

    config_path = OUTPUT_DIR / "yolo_data.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    summary_path = REPORT_DIR / "dataset_quality.json"
    summary_path.write_text(
        json.dumps({
            "source_class_mapping": SOURCE_CLASSES,
            "detection_class_mapping": ANNOTATION_CLASSES,
            "accepted": accepted_summary,
            "rejected_image_count": len(rejected_rows),
            "rejected_reasons": dict(rejected_counts),
            "image_storage_policy": (
                "The generated detection dataset uses hard links, or symbolic links when hard links are unavailable, "
                "instead of copying image data. Original images and label files are not modified."
            ),
            "no_tumor_policy": (
                "No Tumor images are accepted as background examples. Original class-2 boxes are removed from "
                "generated detection labels; class 2 is not a detection class."
            ),
            "multi_class_policy": (
                "An image may contain bounding boxes from more than one tumor class. Source class 2 (No Tumor) "
                "is not allowed in tumor-image annotations; the image folder class must appear among its tumor labels."
            ),
        }, indent=2),
        encoding="utf-8",
    )

    return config_path


if __name__ == "__main__":
    print(create_yolo_config())
