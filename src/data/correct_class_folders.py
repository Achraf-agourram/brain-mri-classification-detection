
import argparse
import math
import shutil
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "mri_classification_corrected"

SPLIT_NAMES = ("Train", "Val")

CLASS_ID_TO_NAME = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_class_ids(label_path):
    try:
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError):
        return None, "invalid_label"

    if not lines:
        return set(), "empty_label"

    class_ids = set()

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            return None, "invalid_label"

        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            return None, "invalid_label"

        if not all(math.isfinite(value) for value in coordinates):
            return None, "invalid_label"

        class_ids.add(class_id)

    if len(class_ids) > 1:
        return class_ids, "multiple_classes"

    return class_ids, None

