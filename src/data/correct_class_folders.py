
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


def collect_images():
    records = []

    for split_name in SPLIT_NAMES:
        split_dir = DATA_DIR / split_name

        for class_name in CLASS_ID_TO_NAME.values():
            image_dir = split_dir / class_name / "images"

            if not image_dir.is_dir():
                continue

            for image_path in sorted(image_dir.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    records.append({
                        "split": split_name,
                        "current_class": class_name,
                        "image_path": image_path,
                        "label_path": (
                            split_dir
                            / class_name
                            / "labels"
                            / f"{image_path.stem}.txt"
                        )
                    })

    return records


def build_plan():
    actions = []
    stats = Counter()
    reserved_targets = set()

    for record in collect_images():
        image_path = record["image_path"]
        label_path = record["label_path"]
        split_name = record["split"]
        current_class = record["current_class"]

        if not label_path.is_file():
            stats["skipped_missing_label"] += 1
            continue

        class_ids, reason = read_class_ids(label_path)

        if reason == "multiple_classes":
            actions.append({
                **record,
                "action": "quarantine"
            })
            stats["ambiguous_images"] += 1
            continue

        if reason == "empty_label":
            stats["skipped_empty_label"] += 1
            continue

        if reason == "invalid_label":
            stats["skipped_invalid_label"] += 1
            continue

        class_id = next(iter(class_ids))

        if class_id not in CLASS_ID_TO_NAME:
            stats["skipped_unknown_class_id"] += 1
            continue

        correct_class = CLASS_ID_TO_NAME[class_id]

        if current_class == correct_class:
            stats["already_correct"] += 1
            continue

        target_dir = DATA_DIR / split_name / correct_class
        target_image = target_dir / "images" / image_path.name
        target_label = target_dir / "labels" / label_path.name

        target_keys = {
            str(target_image.resolve()),
            str(target_label.resolve())
        }

        if (
            target_image.exists()
            or target_label.exists()
            or target_keys & reserved_targets
        ):
            stats["skipped_destination_conflict"] += 1
            continue

        reserved_targets.update(target_keys)

        actions.append({
            **record,
            "action": "reclassify",
            "correct_class": correct_class
        })

        stats["planned_reclassifications"] += 1

    return actions, stats


def create_corrected_copy(actions):
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}. "
            "Rename or remove that directory before creating a new copy."
        )

    for split_name in SPLIT_NAMES:
        source_split = DATA_DIR / split_name

        if not source_split.is_dir():
            raise FileNotFoundError(
                f"Dataset split not found: {source_split}"
            )

    for split_name in SPLIT_NAMES:
        shutil.copytree(
            DATA_DIR / split_name,
            OUTPUT_DIR / split_name
        )

    stats = Counter()

    for action in actions:
        image_path = action["image_path"]
        label_path = action["label_path"]

        copied_image = OUTPUT_DIR / image_path.relative_to(DATA_DIR)
        copied_label = OUTPUT_DIR / label_path.relative_to(DATA_DIR)

        split_name = action["split"]
        current_class = action["current_class"]

        if action["action"] == "quarantine":
            continue

        correct_class = action["correct_class"]
        target_class_dir = (
            OUTPUT_DIR / split_name / correct_class
        )

        target_image = target_class_dir / "images" / image_path.name
        target_label = target_class_dir / "labels" / label_path.name

        if target_image.exists() or target_label.exists():
            stats["skipped_destination_conflict"] += 1
            print(f"SKIPPED CONFLICT: {image_path.name}")
            continue

        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(copied_image), str(target_image))
        shutil.move(str(copied_label), str(target_label))

        stats["reclassified_images"] += 1

        print(
            f"RECLASSIFIED: {image_path.name} "
            f"({current_class} -> {correct_class})"
        )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    actions, plan_stats = build_plan()

    print("Planned reclassifications:")
    for action in actions:
        if action["action"] == "reclassify":
            print(
                f"{action['image_path'].name}: "
                f"{action['current_class']} -> "
                f"{action['correct_class']}"
            )
        else:
            print(
                f"Quarantine multiple-class image: "
                f"{action['image_path'].name}"
            )

    print("\nAnalysis summary:")
    for name, count in sorted(plan_stats.items()):
        print(f"{name}: {count}")

    if not args.apply:
        print("\nPreview only. No files have been copied or changed.")
        print("Use --apply to create the corrected dataset.")
        return

    execution_stats = create_corrected_copy(actions)

    print(f"\nCorrected dataset created at: {OUTPUT_DIR}")

    print("\nExecution summary:")
    for name, count in sorted(execution_stats.items()):
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
