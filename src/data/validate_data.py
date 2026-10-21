from pathlib import Path
from collections import Counter
import math
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
TOLERANCE = 1e-5


def get_image_key(path):
    return path.parent.parent.resolve(), path.stem.casefold()


def get_label_paths():
    return [
        path for path in DATA_DIR.rglob("*.txt")
        if path.is_file() and path.parent.name.lower() == "labels"
    ]


def validate_label_file(path):
    errors = []
    class_ids = set()
    box_count = 0

    with open(path, "r", encoding="utf-8") as file:
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

                if (
                    x_center - width / 2 < -TOLERANCE
                    or y_center - height / 2 < -TOLERANCE
                    or x_center + width / 2 > 1 + TOLERANCE
                    or y_center + height / 2 > 1 + TOLERANCE
                ):
                    raise ValueError("Bounding box outside image")

                class_ids.add(class_id)
                box_count += 1

            except (ValueError, OverflowError) as error:
                errors.append(
                    f"{path}, line {line_number}: {error}: {line.strip()}"
                )

    return errors, class_ids, box_count


image_paths = [
    path for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
    and path.parent.name.lower() == "images"
]

label_paths = get_label_paths()

image_index = {
    get_image_key(path): path
    for path in image_paths
}

label_index = {}

for path in label_paths:
    key = get_image_key(path)

    if key in label_index:
        print("Duplicate annotation filename:", path)

    label_index[key] = path

corrupted_images = []
missing_annotations = []
orphan_annotations = []
invalid_annotations = []
folder_conflicts = []

box_counts = Counter()
class_distribution = Counter()
empty_labels = 0

for image_path in image_paths:
    split_name = image_path.parents[2].name
    folder_class = image_path.parent.parent.name

    class_distribution[(split_name, folder_class)] += 1

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception:
        corrupted_images.append(str(image_path))
        continue

    annotation_path = label_index.get(get_image_key(image_path))

    if annotation_path is None:
        missing_annotations.append(str(image_path))
        continue

    errors, class_ids, box_count = validate_label_file(annotation_path)

    invalid_annotations.extend(errors)

    if box_count == 0:
        empty_labels += 1

    for class_id in class_ids:
        box_counts[CLASS_NAMES[class_id]] += 1

    if folder_class in CLASS_NAMES:
        expected_class_id = CLASS_NAMES.index(folder_class)

        if any(class_id != expected_class_id for class_id in class_ids):
            folder_conflicts.append(str(image_path))

for key, label_path in label_index.items():
    if key not in image_index:
        orphan_annotations.append(str(label_path))

print("Total images:", len(image_paths))
print("Total annotation files:", len(label_paths))
print("Corrupted images:", len(corrupted_images))
print("Images without annotations:", len(missing_annotations))
print("Annotations without images:", len(orphan_annotations))
print("Invalid annotation rows:", len(invalid_annotations))
print("Empty annotation files:", empty_labels)
print("Folder/annotation class disagreements:", len(folder_conflicts))

print("\nImage distribution:")

for (split_name, class_name), count in sorted(class_distribution.items()):
    print(f"{split_name} / {class_name}: {count}")

print("\nBounding boxes by annotation class:")

for class_name in CLASS_NAMES:
    print(f"{class_name}: {box_counts[class_name]}")

print("\nCorrupted images:")
print(*corrupted_images, sep="\n")

print("\nImages without annotations:")
print(*missing_annotations, sep="\n")

print("\nAnnotations without images:")
print(*orphan_annotations, sep="\n")

print("\nInvalid annotation rows:")
print(*invalid_annotations, sep="\n")

print("\nFolder/annotation class disagreements:")
print(*folder_conflicts, sep="\n")