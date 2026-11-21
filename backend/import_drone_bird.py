import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml


SPLITS = {"train": "train", "valid": "val", "test": "test"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_SOURCE_NAMES = ["bird", "drone"]


def inspect_source(source: Path):
    config_path = source / "data.yaml"
    if not config_path.is_file():
        raise ValueError("Source data.yaml is missing")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = config.get("names")

    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    if not isinstance(names, list):
        raise ValueError("Invalid source class names")

    if [str(n).strip().lower() for n in names] != EXPECTED_SOURCE_NAMES:
        raise ValueError(f"Unexpected source classes: {names}")

    if config.get("nc") != 2:
        raise ValueError("Expected exactly two source classes")

    return config


def validate_annotation(path: Path):
    converted = []
    counts = Counter()

    for number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = line.split()

        if not fields:
            continue

        if fields[0] not in ("0", "1"):
            raise ValueError(f"Unexpected class: {path}:{number}")

        try:
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(
                f"Invalid coordinates: {path}:{number}"
            ) from exc

        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite values: {path}:{number}")

        if len(values) == 4:
            x, y, w, h = values

            if not (
                0 <= x <= 1 and 0 <= y <= 1
                and 0 < w <= 1 and 0 < h <= 1
                and x - w / 2 >= -1e-6
                and x + w / 2 <= 1 + 1e-6
                and y - h / 2 >= -1e-6
                and y + h / 2 <= 1 + 1e-6
            ):
                raise ValueError(f"Invalid bounding box: {path}:{number}")

        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]

            if not all(0 <= value <= 1 for value in values):
                raise ValueError(
                    f"Polygon coordinates outside image: {path}:{number}"
                )

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            w = xmax - xmin
            h = ymax - ymin

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Degenerate polygon: {path}:{number}"
                )

            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2

        else:
            raise ValueError(
                f"Unsupported annotation format: {path}:{number}"
            )

        source_id = int(fields[0])
        target_id = 1 - source_id

        counts["drone" if target_id == 0 else "bird"] += 1

        converted.append(
            f"{target_id} {x:.10f} {y:.10f} {w:.10f} {h:.10f}"
        )

    return converted, counts



def prepare(source: Path, destination: Path):
    source = source.resolve()
    destination = destination.resolve()

    if destination.exists():
        raise ValueError("Destination already exists; refusing overwrite")

    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside the source")

    config = inspect_source(source)

    audit_dir = source.parent
    iou_path = audit_dir / "annotation_iou_report.json"

    if not iou_path.is_file():
        raise ValueError("Missing annotation IoU audit report")

    iou_report = json.loads(iou_path.read_text(encoding="utf-8"))

    # Verify audited source files before import.
    plan_path = source.parent / "split_plan.json"
    if not plan_path.is_file():
        raise ValueError("Missing mandatory split audit plan")

    from audit_integrity import verify_audit
    integrity = verify_audit(source)
    print("Audit integrity verified:", integrity, flush=True)

    verified_pairs = set()

    for result in iou_report.get("results", []):
        minimum = result.get("min_iou")

        if (
            minimum is None
            or minimum < 0.80
            or result.get("status") not in (
                "very_high_match", "good_match"
            )
        ):
            raise ValueError("Unresolved annotation conflict in IoU report")

        left = (result["left_split"], result["left_image"])
        right = (result["right_split"], result["right_image"])

        verified_pairs.add(frozenset((left, right)))

    prepared = []
    by_hash = {}
    totals_source = Counter()
    source_counts = Counter()
    empty_labels = 0
    errors = []

    print("Preflight: validating source dataset...", flush=True)

    for source_split, target_split in SPLITS.items():
        image_dir = source / source_split / "images"
        label_dir = source / source_split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"Missing split directories: {source_split}")

        images = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        labels = {
            p.relative_to(label_dir).with_suffix(""): p
            for p in label_dir.rglob("*.txt")
        }

        image_keys = [
            p.relative_to(image_dir).with_suffix("")
            for p in images
        ]

        if not images:
            errors.append(f"Empty split: {source_split}")

        if len(set(image_keys)) != len(image_keys):
            errors.append(f"Ambiguous image stems: {source_split}")

        for orphan in sorted(set(labels) - set(image_keys)):
            errors.append(f"Orphan label: {source_split}/{orphan}")

        source_counts[target_split] = len(images)

        for image in images:
            relative = image.relative_to(image_dir)
            key = relative.with_suffix("")
            label = labels.get(key)

            if label is None:
                errors.append(f"Missing label: {image}")
                continue

            decoded = cv2.imread(str(image))

            if decoded is None or decoded.size == 0:
                errors.append(f"Unreadable image: {image}")
                continue

            try:
                converted, counts = validate_annotation(label)
            except (ValueError, UnicodeError) as exc:
                errors.append(str(exc))
                continue

            digest = hashlib.sha256(image.read_bytes()).hexdigest()

            item = {
                "image": image,
                "relative": relative,
                "source_split": source_split,
                "target_split": target_split,
                "converted": converted,
                "counts": counts,
                "digest": digest,
            }

            prepared.append(item)
            by_hash.setdefault(digest, []).append(item)

            totals_source.update(counts)

            if not converted:
                empty_labels += 1

    if errors:
        raise ValueError(
            f"Preflight failed with {len(errors)} errors: "
            + "; ".join(errors[:10])
        )

    priority = {"test": 0, "val": 1, "train": 2}

    selected = []
    excluded = []
    verified_groups = 0

    for digest, group in sorted(by_hash.items()):
        ordered = sorted(
            group,
            key=lambda item: (
                priority[item["target_split"]],
                item["relative"].as_posix(),
            ),
        )

        # Every cross-split pair must have a verified audit entry.
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                if left["target_split"] == right["target_split"]:
                    continue

                pair = frozenset((
                    (left["source_split"], left["image"].name),
                    (right["source_split"], right["image"].name),
                ))

                if pair not in verified_pairs:
                    raise ValueError(
                        "Missing verified annotation comparison for "
                        f"{left['image']} and {right['image']}"
                    )

                if Counter(
                    line.split()[0] for line in left["converted"]
                ) != Counter(
                    line.split()[0] for line in right["converted"]
                ):
                    raise ValueError(
                        "Duplicate images have conflicting object classes"
                    )

        if len({item["target_split"] for item in group}) > 1:
            verified_groups += 1

        selected.append(ordered[0])
        excluded.extend(ordered[1:])

    expected_groups = iou_report.get("total_duplicate_groups")

    if verified_groups != expected_groups:
        raise ValueError(
            "Duplicate-group count differs from annotation audit"
        )

    selected_counts = Counter(
        item["target_split"] for item in selected
    )

    excluded_counts = Counter(
        item["target_split"] for item in excluded
    )

    selected_annotations = Counter()

    for item in selected:
        selected_annotations.update(item["counts"])

    report = {
        "source": str(source),
        "destination": str(destination),
        "source_images_by_split": dict(source_counts),
        "prepared_images_by_split": dict(selected_counts),
        "excluded_copies_by_split": dict(excluded_counts),
        "excluded_copies_total": len(excluded),
        "verified_cross_split_duplicate_groups": verified_groups,
        "source_annotations_by_class": dict(totals_source),
        "prepared_annotations_by_class": dict(selected_annotations),
        "source_empty_label_files": empty_labels,
        "source_metadata": config.get("roboflow", {}),
        "source_images_modified": False,
        "source_images_deleted": False,
        "excluded_copies": [
            {
                "source_split": item["source_split"],
                "image": item["relative"].as_posix(),
                "sha256": item["digest"],
            }
            for item in excluded
        ],
    }

    print(json.dumps({
        key: value for key, value in report.items()
        if key != "excluded_copies"
    }, indent=2), flush=True)

    destination.mkdir(parents=True)

    try:
        for index, item in enumerate(selected, 1):
            split = item["target_split"]
            relative = item["relative"]

            output_image = destination / split / "images" / relative
            output_label = (
                destination / split / "labels"
                / relative.with_suffix(".txt")
            )

            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(item["image"], output_image)

            # Verify exact image-byte preservation.
            copied_hash = hashlib.sha256(
                output_image.read_bytes()
            ).hexdigest()

            if copied_hash != item["digest"]:
                raise ValueError(f"Image copy mismatch: {output_image}")

            converted = item["converted"]

            output_label.write_text(
                "\n".join(converted)
                + ("\n" if converted else ""),
                encoding="utf-8",
            )

            if index % 1000 == 0:
                print(
                    f"Imported {index}/{len(selected)} images",
                    flush=True,
                )

        dataset_yaml = {
            "path": str(destination),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 2,
            "names": {0: "drone", 1: "bird"},
        }

        (destination / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False),
            encoding="utf-8",
        )

        (destination / "import_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    except Exception:
        shutil.rmtree(destination)
        raise

    print("IMPORT COMPLETED", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    prepare(Path(args.source), Path(args.destination))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"IMPORT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
