from pathlib import Path
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
SAMPLE_COUNT = 12
TOLERANCE = 1e-5


def get_image_paths():
    return [
        path for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_annotation_path(image_path):
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def get_label_paths():
    return [
        path for path in DATA_DIR.rglob("*.txt")
        if path.is_file() and path.parent.name.lower() == "labels"
    ]


image_paths = get_image_paths()
label_paths = get_label_paths()

image_keys = {
    (path.parent.parent, path.stem)
    for path in image_paths
}

label_keys = {
    (path.parent.parent, path.stem)
    for path in label_paths
}

corrupted_images = []
missing_annotations = []
orphan_annotations = []
invalid_annotations = []

for image_path in image_paths:
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception:
        corrupted_images.append(str(image_path))
        continue

    annotation_path = get_annotation_path(image_path)

    if not annotation_path.is_file():
        missing_annotations.append(str(image_path))
        continue

    with open(annotation_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.split()

            try:
                if len(values) != 5:
                    raise ValueError("Expected five values")

                class_value = float(values[0])

                if not class_value.is_integer():
                    raise ValueError("Invalid class ID")

                class_id = int(class_value)
                x_center, y_center, width, height = map(float, values[1:])

                if not 0 <= class_id < len(CLASS_NAMES):
                    raise ValueError("Unknown class ID")

                if not all(
                    -TOLERANCE <= value <= 1 + TOLERANCE
                    for value in [x_center, y_center, width, height]
                ):
                    raise ValueError("Invalid normalized coordinates")

                if width <= 0 or height <= 0:
                    raise ValueError("Invalid bounding box size")

                if (
                    x_center - width / 2 < -TOLERANCE
                    or y_center - height / 2 < -TOLERANCE
                    or x_center + width / 2 > 1 + TOLERANCE
                    or y_center + height / 2 > 1 + TOLERANCE
                ):
                    raise ValueError("Bounding box outside image")

            except (ValueError, OverflowError):
                invalid_annotations.append(
                    f"{annotation_path}, line {line_number}: {line.strip()}"
                )

for label_path in label_paths:
    key = (label_path.parent.parent, label_path.stem)

    if key not in image_keys:
        orphan_annotations.append(str(label_path))

print("Total images:", len(image_paths))
print("Total label files:", len(label_paths))
print("Corrupted images:", len(corrupted_images))
print("Images without annotations:", len(missing_annotations))
print("Annotations without images:", len(orphan_annotations))
print("Invalid annotations:", len(invalid_annotations))

print("\nImages without annotations:")
print(*missing_annotations, sep="\n")

print("\nAnnotations without images:")
print(*orphan_annotations, sep="\n")

print("\nCorrupted images:")
print(*corrupted_images, sep="\n")

print("\nInvalid annotations:")
print(*invalid_annotations, sep="\n")

valid_images = [
    path for path in image_paths
    if str(path) not in corrupted_images
]

samples = random.sample(valid_images, min(SAMPLE_COUNT, len(valid_images)))

if samples:
    columns = 4
    rows = (len(samples) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, image_path in zip(axes, samples):
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        draw = ImageDraw.Draw(image)
        annotation_path = get_annotation_path(image_path)

        if annotation_path.is_file():
            with open(annotation_path, "r", encoding="utf-8") as file:
                for line in file:
                    values = line.split()

                    try:
                        if len(values) != 5:
                            continue

                        class_value = float(values[0])

                        if not class_value.is_integer():
                            continue

                        class_id = int(class_value)
                        x_center, y_center, width, height = map(float, values[1:])

                        if not 0 <= class_id < len(CLASS_NAMES):
                            continue

                        image_width, image_height = image.size

                        x1 = (x_center - width / 2) * image_width
                        y1 = (y_center - height / 2) * image_height
                        x2 = (x_center + width / 2) * image_width
                        y2 = (y_center + height / 2) * image_height

                        draw.rectangle(
                            (x1, y1, x2, y2),
                            outline="red",
                            width=2
                        )

                        draw.text(
                            (max(0, x1), max(0, y1 - 15)),
                            CLASS_NAMES[class_id],
                            fill="yellow"
                        )

                    except (ValueError, OverflowError):
                        continue

        ax.imshow(image)
        ax.set_title(image_path.relative_to(DATA_DIR).as_posix())
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()