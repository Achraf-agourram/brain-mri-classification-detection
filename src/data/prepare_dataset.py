from pathlib import Path
from collections import Counter
import csv
import hashlib
import math
import random
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
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

