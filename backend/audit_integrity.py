import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_audit(source):
    source = Path(source).resolve()
    incoming = source.parent

    plan = json.loads(
        (incoming / "split_plan.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (incoming / "annotation_iou_report.json").read_text(
            encoding="utf-8"
        )
    )

    decisions = plan["decisions"]
    results = audit["results"]

    if len(decisions) != plan["duplicate_groups"]:
        raise ValueError("Invalid duplicate-group count in split plan")

    if len(results) != audit["total_comparisons"]:
        raise ValueError("Invalid IoU comparison count")

    expected_pairs = set()
    excluded_counts = Counter()

    for decision in decisions:
        selected = decision["selected"]
        excluded = decision["excluded_from_prepared_dataset"]

        if not excluded:
            raise ValueError("Decision has no excluded copies")

        entries = [selected, *excluded]
        image_hashes = []

        for entry in entries:
            split = entry["split"]
            image_name = entry["image"]

            if split not in {"train", "valid", "test"}:
                raise ValueError("Invalid split")

            if Path(image_name).name != image_name:
                raise ValueError("Invalid image name")

            expected_label = (
                Path(split) / "labels"
                / (Path(image_name).stem + ".txt")
            )

            actual_label = Path(
                entry["label"].replace("\\", "/")
            )

            if actual_label != expected_label:
                raise ValueError("Unexpected label path")

            label_path = source / actual_label
            image_path = source / split / "images" / image_name

            if not label_path.is_file() or not image_path.is_file():
                raise ValueError("Missing audited source file")

            if sha256_file(label_path) != entry["label_hash"]:
                raise ValueError("Annotation changed after audit")

            image_hashes.append(sha256_file(image_path))

        if len(set(image_hashes)) != 1:
            raise ValueError("Duplicate image bytes no longer match")

        for item in excluded:
            excluded_counts[item["split"]] += 1

            expected_pairs.add(frozenset((
                (selected["split"], selected["image"]),
                (item["split"], item["image"]),
            )))

    actual_pairs = set()

    for result in results:
        minimum = result.get("min_iou")

        if (
            not isinstance(minimum, (int, float))
            or not 0.80 <= minimum <= 1.0
            or result.get("status")
            not in {"very_high_match", "good_match"}
        ):
            raise ValueError("Unresolved annotation comparison")

        actual_pairs.add(frozenset((
            (result["left_split"], result["left_image"]),
            (result["right_split"], result["right_image"]),
        )))

    if expected_pairs != actual_pairs:
        raise ValueError("Split plan and IoU audit disagree")

    if len(actual_pairs) != len(results):
        raise ValueError("Repeated IoU comparison")

    if sum(excluded_counts.values()) != plan["excluded_copies_total"]:
        raise ValueError("Incorrect excluded-copy count")

    if dict(excluded_counts) != plan["excluded_copies_by_split"]:
        raise ValueError("Incorrect excluded-copy distribution")

    return {
        "verified_groups": len(decisions),
        "verified_pairs": len(actual_pairs),
        "excluded_copies": sum(excluded_counts.values()),
        "annotation_hashes_verified": True,
        "duplicate_image_hashes_verified": True,
    }