"""Compare an image analysis against independent, manually annotated XYXY boxes."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

from app.api.media_upload import _model, _model_path, _summary, _verify_image_birds
from skyguard_fix.cli import evaluate_file, sha
from skyguard_fix.core import Detection, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    if sha(args.image) != annotation["source_sha256"]:
        parser.error("The annotated image hash does not match")
    report = evaluate_file(args.image, args.output, _model(), image_verifier=_verify_image_birds)
    summary = _summary(args.output / "detections.jsonl", image=True)
    measured = summary["detections"]
    truth = annotation["manual_approximate_annotation_xyxy"]
    best = {}
    for label, box in truth.items():
        target = Detection(label, 1., tuple(box), "annotation")
        best[label] = max((
            overlap(target, Detection(label, 1., tuple(row["bbox"]), "inference"))
            for row in measured if row["classification"] == label
        ), default=0.)
    passed = (
        summary["counts"] == annotation["expected_counts"] and
        all(value >= annotation["minimum_iou"] for value in best.values())
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "checkpoint_sha256": sha(_model_path()),
        "ultralytics_version": version("ultralytics"),
        "checkpoint_classes": report["model_labels"],
        "expected_counts": annotation["expected_counts"],
        "observed_counts": summary["counts"],
        "best_same_label_iou": best,
        "minimum_iou": annotation["minimum_iou"],
        "unknown_trained": report["v2_has_unknown"],
        "detections": measured,
    }
    encoded = json.dumps(result, indent=2)
    (args.output / "validation.json").write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
