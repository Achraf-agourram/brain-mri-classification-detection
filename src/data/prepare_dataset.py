from pathlib import Path
from collections import Counter
import csv
import hashlib
import math
import random
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed" / "mri_classification_corrected"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "mri_dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
SEED = 42
VALIDATION_RATIO = 0.10
TOLERANCE = 1e-5


def get_pixel_hash(image_path):
    with Image.open(image_path) as image:
        image.load()
        image = image.convert("RGB")
        content = f"{image.width}x{image.height}".encode() + image.tobytes()

    return hashlib.sha256(content).hexdigest()


def parse_annotations(label_path):
    rows = []

    if label_path is None:
        return rows

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.split()

            if not values:
                continue

            try:
                if len(values) != 5:
                    raise ValueError("Expected five values")

                class_value = float(values[0])

                if not math.isfinite(class_value) or not class_value.is_integer():
                    raise ValueError("Invalid class ID")

                class_id = int(class_value)
                x_center, y_center, width, height = map(float, values[1:])

                if not 0 <= class_id < len(CLASS_NAMES):
                    raise ValueError("Unknown class ID")

                if not all(
                    math.isfinite(value)
                    for value in [x_center, y_center, width, height]
                ):
                    raise ValueError("Non-finite coordinate")

                if not (
                    -TOLERANCE <= x_center <= 1 + TOLERANCE
                    and -TOLERANCE <= y_center <= 1 + TOLERANCE
                    and 0 < width <= 1 + TOLERANCE
                    and 0 < height <= 1 + TOLERANCE
                ):
                    raise ValueError("Invalid normalized coordinates")

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
                    raise ValueError("Bounding box outside image")

                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(1.0, x2)
                y2 = min(1.0, y2)

                rows.append((
                    class_id,
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1
                ))

            except (ValueError, OverflowError) as error:
                raise ValueError(
                    f"{label_path}, line {line_number}: {error}"
                ) from error

    return rows


def get_records():
    records = []
    orphan_labels = []
    missing_labels = []

    for source_split in ["Train", "Val"]:
        split_dir = DATA_DIR / source_split

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {split_dir}")

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"

            if not images_dir.is_dir():
                raise FileNotFoundError(f"Missing directory: {images_dir}")

            if not labels_dir.is_dir():
                raise FileNotFoundError(f"Missing directory: {labels_dir}")

            label_index = {}

            for label_path in labels_dir.rglob("*"):
                if not label_path.is_file() or label_path.suffix.lower() != ".txt":
                    continue

                key = label_path.stem.casefold()

                if key in label_index:
                    raise ValueError(
                        f"Duplicate label filename: {label_path}"
                    )

                label_index[key] = label_path

            image_paths = sorted(
                path for path in images_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            image_stems = {path.stem.casefold() for path in image_paths}

            for label_path in label_index.values():
                if label_path.stem.casefold() not in image_stems:
                    orphan_labels.append(str(label_path))

            for image_path in image_paths:
                label_path = label_index.get(image_path.stem.casefold())

                if label_path is None and class_name != "No Tumor":
                    missing_labels.append(str(image_path))
                    continue

                try:
                    pixel_hash = get_pixel_hash(image_path)
                    rows = parse_annotations(label_path)
                except Exception as error:
                    raise ValueError(
                        f"Cannot process {image_path}: {error}"
                    ) from error

                annotation_classes = sorted({row[0] for row in rows})

                records.append({
                    "source_split": source_split,
                    "source_path": image_path.relative_to(DATA_DIR).as_posix(),
                    "image_path": image_path.resolve().as_posix(),
                    "label_path": (
                        label_path.resolve().as_posix()
                        if label_path is not None
                        else ""
                    ),
                    "folder_class": class_name,
                    "class_id": CLASS_NAMES.index(class_name),
                    "pixel_hash": pixel_hash,
                    "rows": rows,
                    "annotation_classes": annotation_classes
                })

    if missing_labels:
        raise ValueError(
            "Images outside No Tumor have no annotations:\n"
            + "\n".join(missing_labels[:20])
        )

    return records, orphan_labels


def remove_duplicates(records):
    kept_records = []
    seen_hashes = {}
    dropped_records = []

    records.sort(key=lambda record: (
        record["source_split"] != "Val",
        record["source_path"]
    ))

    for record in records:
        pixel_hash = record["pixel_hash"]

        if pixel_hash in seen_hashes:
            previous = seen_hashes[pixel_hash]

            dropped_records.append(record["source_path"])
            continue

        seen_hashes[pixel_hash] = record
        kept_records.append(record)

    return kept_records, dropped_records


def create_splits(records):
    test_records = [
        record for record in records
        if record["source_split"] == "Val"
    ]

    training_records = [
        record for record in records
        if record["source_split"] == "Train"
    ]

    rng = random.Random(SEED)
    train_records = []
    validation_records = []

    for class_name in CLASS_NAMES:
        class_records = [
            record for record in training_records
            if record["folder_class"] == class_name
        ]

        rng.shuffle(class_records)

        validation_count = (
            max(1, round(len(class_records) * VALIDATION_RATIO))
            if len(class_records) >= 2
            else 0
        )

        validation_records.extend(class_records[:validation_count])
        train_records.extend(class_records[validation_count:])

    splits = {
        "train": train_records,
        "validation": validation_records,
        "test": test_records
    }

    split_hashes = {
        name: {record["pixel_hash"] for record in split_records}
        for name, split_records in splits.items()
    }

    if (
        split_hashes["train"] & split_hashes["validation"]
        or split_hashes["train"] & split_hashes["test"]
        or split_hashes["validation"] & split_hashes["test"]
    ):
        raise ValueError("Duplicate images detected across splits")

    return splits


def create_output_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_splits(splits, orphan_labels, dropped_records):
    create_output_directories()

    fields = [
        "image_path",
        "label_path",
        "class_name",
        "class_id",
        "split",
        "source_split",
        "annotation_classes",
        "annotation_status",
        "folder_label_conflict",
        "pixel_hash"
    ]

    manifest_rows = []
    summary = Counter()

    for split_name, records in splits.items():
        csv_path = OUTPUT_DIR / f"{split_name}.csv"
        txt_path = OUTPUT_DIR / f"{split_name}.txt"
        split_rows = []

        for record in records:
            class_name = record["folder_class"]
            annotation_classes = record["annotation_classes"]
            folder_class_id = record["class_id"]

            folder_label_conflict = any(
                class_id != folder_class_id
                for class_id in annotation_classes
            )

            if not record["label_path"]:
                annotation_status = "missing_assumed_background"
            elif not record["rows"]:
                annotation_status = "empty_label"
            else:
                annotation_status = "labeled"

            row = {
                "image_path": record["image_path"],
                "label_path": record["label_path"],
                "class_name": class_name,
                "class_id": folder_class_id,
                "split": split_name,
                "source_split": record["source_split"],
                "annotation_classes": ";".join(
                    CLASS_NAMES[class_id]
                    for class_id in annotation_classes
                ),
                "annotation_status": annotation_status,
                "folder_label_conflict": folder_label_conflict,
                "pixel_hash": record["pixel_hash"]
            }

            split_rows.append(row)
            manifest_rows.append(row)
            summary[(split_name, class_name)] += 1

        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows)

    with open(
        OUTPUT_DIR / "manifest.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(
        OUTPUT_DIR / "split_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["split", "class", "image_count"])

        for (split_name, class_name), count in sorted(summary.items()):
            writer.writerow([split_name, class_name, count])

    print("\nDataset splits created.")

    for split_name, records in splits.items():
        print(f"{split_name.capitalize()} images: {len(records)}")

    print("Duplicates excluded:", len(dropped_records))
    print("Orphan annotation files:", len(orphan_labels))
    print("Output directory:", OUTPUT_DIR)


def main():
    records, orphan_labels = get_records()

    print("Images loaded:", len(records))
    print("Orphan annotation files:", len(orphan_labels))

    records, dropped_records = remove_duplicates(records)

    print("Images after duplicate removal:", len(records))
    print("Duplicate images excluded:", len(dropped_records))

    splits = create_splits(records)
    write_splits(splits, orphan_labels, dropped_records)


if __name__ == "__main__":
    main()