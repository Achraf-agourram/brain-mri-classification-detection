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

