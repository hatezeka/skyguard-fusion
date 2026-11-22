"""Score a small deterministic annotated sample with the operator's real pipeline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from skyguard_fix.core import UltralyticsPredictor, multiscale
from skyguard_multiscale.detection import tier_for
from skyguard_multiscale.evaluate import evaluate_frame, load_yolo_labels, merge_stats


def source_stem(path):
    return path.name.lower().split(".rf.", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--exclude-shared-stems", action="store_true",
                        help="Exclude test files whose exported source stem appears in sibling train or val")
    args = parser.parse_args()
    excluded = set()
    if args.exclude_shared_stems:
        for split in ("train", "val"):
            split_images = args.dataset.parent / split / "images"
            excluded.update(source_stem(path) for path in split_images.iterdir() if path.is_file())
    images = {file.stem: file for file in (args.dataset / "images").iterdir() if file.is_file()}
    files = sorted((args.dataset / "labels").glob("*.txt"))
    chosen = Counter()
    samples = []
    predictor = UltralyticsPredictor(args.weights)
    for label_file in files:
        lines = label_file.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1:
            continue
        label = lines[0].split()[0]
        if label not in ("0", "1") or chosen[label] >= args.per_class or label_file.stem not in images:
            continue
        path = images[label_file.stem]
        if source_stem(path) in excluded:
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode {path}")
        height, width = image.shape[:2]
        targets = load_yolo_labels(label_file, width, height)
        dets = multiscale(image, predictor, predictor.names, include_tiles=True)
        predictions = [
            dict(class_name=d.label, xyxy=d.xyxy, confidence=d.confidence, tier=tier_for(d.xyxy, width, height))
            for d in dets if d.verdict_status != "CONFLICT_REVIEW"
        ]
        scores = evaluate_frame(predictions, targets)
        samples.append(dict(image=path.name, class_name=targets[0]["class_name"], scores=scores))
        chosen[label] += 1
        if sum(chosen.values()) == 2 * args.per_class:
            break
    if len(chosen) != 2 or any(count != args.per_class for count in chosen.values()):
        raise ValueError("Insufficient annotated samples for both classes")
    combined = merge_stats(sample["scores"] for sample in samples)
    class_reports = [{key.split("/")[0]: value} for key, value in combined.items()]
    with args.weights.open("rb") as model:
        sha = hashlib.file_digest(model, "sha256").hexdigest()
    result = dict(checkpoint_sha256=sha, sample_size=len(samples), iou_threshold=0.5,
                  shared_source_stems_excluded=args.exclude_shared_stems,
                  sampling="First sorted single-object annotation files per class in the supplied test partition",
                  limitation="Selected test sample. Source disjointness and field accuracy were not independently established.",
                  by_class=merge_stats(class_reports), by_class_and_size=combined, samples=samples)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
