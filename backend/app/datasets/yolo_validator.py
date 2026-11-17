from dataclasses import dataclass
from pathlib import Path
import hashlib
import math


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetValidationReport:
    images: int
    labels: int
    annotations: int
    duplicate_groups: int
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_dataset(root: str | Path) -> DatasetValidationReport:
    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(root)

    errors = []
    image_count = 0
    label_count = 0
    annotation_count = 0
    duplicate_groups = 0
    seen_hashes = {}

    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"

        if not images_dir.is_dir():
            errors.append(f"Missing images directory: {split}")
            continue

        if not labels_dir.is_dir():
            errors.append(f"Missing labels directory: {split}")
            continue

        images = sorted(
            path for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        labels = sorted(labels_dir.rglob("*.txt"))

        image_count += len(images)
        label_count += len(labels)

        if not images:
            errors.append(f"Empty image split: {split}")

        image_keys = {
            path.relative_to(images_dir).with_suffix("")
            for path in images
        }

        label_keys = {
            path.relative_to(labels_dir).with_suffix("")
            for path in labels
        }

        for key in sorted(label_keys - image_keys):
            errors.append(f"Orphan label: {split}/{key}")

        for image in images:
            key = image.relative_to(images_dir).with_suffix("")
            label_path = labels_dir / key.with_suffix(".txt")

            digest = hashlib.sha256(image.read_bytes()).hexdigest()

            previous = seen_hashes.get(digest)

            if previous is not None and previous[0] != split:
                duplicate_groups += 1
                errors.append(
                    f"Cross-split duplicate: {previous[1]} and {image}"
                )
            else:
                seen_hashes.setdefault(digest, (split, image))

            if not label_path.is_file():
                errors.append(f"Missing label: {split}/{key}")
                continue

            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                fields = line.split()

                if len(fields) != 5:
                    errors.append(
                        f"Invalid annotation fields: {label_path}:{line_number}"
                    )
                    continue

                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(
                        f"Invalid annotation values: {label_path}:{line_number}"
                    )
                    continue

                if class_id not in (0, 1):
                    errors.append(
                        f"Invalid class: {label_path}:{line_number}"
                    )
                    continue

                x, y, width, height = values

                if not all(math.isfinite(value) for value in values):
                    errors.append(
                        f"Non-finite annotation: {label_path}:{line_number}"
                    )
                    continue

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                    and x - width / 2 >= -1e-6
                    and x + width / 2 <= 1 + 1e-6
                    and y - height / 2 >= -1e-6
                    and y + height / 2 <= 1 + 1e-6
                ):
                    errors.append(
                        f"Invalid bounding box: {label_path}:{line_number}"
                    )
                    continue

                annotation_count += 1

    return DatasetValidationReport(
        images=image_count,
        labels=label_count,
        annotations=annotation_count,
        duplicate_groups=duplicate_groups,
        errors=tuple(errors),
    )
