from pathlib import Path
from collections import Counter
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

image_paths = [
    path for path in DATA_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
]

annotation_paths = [
    path for path in DATA_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() == ".txt"
]

print("Dataset directory:", DATA_DIR)
print("Total images:", len(image_paths))
print("Total text annotation files:", len(annotation_paths))

print("\nDataset structure:")

for folder in sorted(path for path in DATA_DIR.rglob("*") if path.is_dir()):
    print(folder.relative_to(DATA_DIR))

print("\nImage distribution by folder:")

folder_counts = Counter(
    path.parent.relative_to(DATA_DIR).as_posix()
    for path in image_paths
)

print(pd.Series(folder_counts).sort_index())

print("\nAnnotation file examples:")

for path in annotation_paths[:5]:
    print(f"\n{path.relative_to(DATA_DIR)}")

    with open(path, "r", encoding="utf-8") as file:
        for line in file.readlines()[:5]:
            print(line.rstrip())

print("\nImage distribution by class folder:")

class_counts = {}

for class_name in CLASS_NAMES:
    class_counts[class_name] = sum(
        class_name.lower() in {
            part.lower() for part in path.relative_to(DATA_DIR).parts
        }
        for path in image_paths
    )

print(pd.Series(class_counts))