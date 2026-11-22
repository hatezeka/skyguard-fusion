import json
import sys
import shutil
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from collections import Counter

import cv2
from mcos_decoder import load_groundtruth


def validate_sequence(archive, row):
    decoded = 0
    annotated = None
    cap = None

    try:
        if not row.get("label_zip_entry"):
            raise ValueError("MISSING_GT")

        with TemporaryDirectory(prefix="skyguard-preflight-") as temp:
            video = Path(temp) / "video.mp4"
            labels = Path(temp) / "labels.mat"

            with archive.open(row["video_zip_entry"]) as src:
                with video.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

            with archive.open(row["label_zip_entry"]) as src:
                with labels.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

            bboxes = load_groundtruth(str(labels))

            if not isinstance(bboxes, (list, tuple)):
                raise ValueError("INVALID_ANNOTATION_FORMAT")

            annotated = len(bboxes)

            try:
                cap = cv2.VideoCapture(str(video))

                if not cap.isOpened():
                    raise ValueError("VIDEO_DECODE_FAILED")

                while True:
                    ok, _ = cap.read()

                    if not ok:
                        break

                    decoded += 1

            finally:
                # IMPORTANT: release before TemporaryDirectory cleanup
                if cap is not None:
                    cap.release()
                    cap = None

            if decoded == 0:
                raise ValueError("EMPTY_VIDEO")

            if decoded != annotated:
                raise ValueError(
                    f"FRAME_MISMATCH decoded={decoded} "
                    f"annotated={annotated} "
                    f"difference={decoded - annotated}"
                )

        # The temporary directory was cleaned successfully.
        return True, {
            "decoded_frames": decoded,
            "annotation_frames": annotated,
        }

    except Exception as exc:
        return False, {
            "decoded_frames": decoded,
            "annotation_frames": annotated,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def main():
    manifest_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    if out_dir.exists():
        raise FileExistsError(out_dir)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = data.get("halmstad", data)

    records = manifest["records"]
    accepted = []
    rejected = []

    with ZipFile(manifest["source_zip_path"]) as archive:
        for index, row in enumerate(records, 1):
            ok, details = validate_sequence(archive, row)

            if ok:
                accepted.append(row)
            else:
                rejected.append({
                    "sequence_id": row["sequence_id"],
                    "class": row["class"],
                    "split": row["split"],
                    "modality": row["modality"],
                    **details,
                })

            if index % 25 == 0 or index == len(records):
                print(
                    f"Checked {index}/{len(records)} | "
                    f"accepted={len(accepted)} | "
                    f"rejected={len(rejected)}",
                    flush=True,
                )

    total = len(records)

    if len(accepted) + len(rejected) != total:
        raise AssertionError("ACCEPTED_REJECTED_INVARIANT_FAILED")

    accepted_ids = {r["sequence_id"] for r in accepted}
    rejected_ids = {r["sequence_id"] for r in rejected}

    if accepted_ids & rejected_ids:
        raise AssertionError("ACCEPTED_REJECTED_OVERLAP")

    if len(accepted_ids | rejected_ids) != total:
        raise AssertionError("DUPLICATE_OR_MISSING_SEQUENCE_IDS")

    counts = Counter(
        (r["split"], r["class"], r["modality"])
        for r in accepted
    )

    clean = dict(manifest)
    clean["records"] = accepted

    report = {
        "status": "PREFLIGHT_COMPLETED_NOT_TRAINED",
        "source_sha256": manifest["source_sha256"],
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_groups": [
            {
                "split": split,
                "class": cls,
                "modality": modality,
                "sequences": count,
            }
            for (split, cls, modality), count in sorted(counts.items())
        ],
        "rejection_reasons": dict(Counter(
            r["reason"].split(":", 1)[0]
            for r in rejected
        )),
        "rejections": rejected,
        "limitations": [
            "Frame-count equality does not prove annotation alignment",
            "Bounding boxes have not been fully validated",
            "Cross-split leakage has not been checked",
            "No model training or accuracy has been established",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=False)

    (out_dir / "clean_manifest.json").write_text(
        json.dumps(clean, indent=2), encoding="utf-8"
    )

    (out_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "status": report["status"],
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_groups": report["accepted_groups"],
        "rejection_reasons": report["rejection_reasons"],
        "output": str(out_dir),
    }, indent=2))

    # Do not silently declare an incomplete dataset train-ready.
    required = {"DRONE", "BIRD", "AIRPLANE", "HELICOPTER"}
    splits = {"train", "val", "test"}

    missing = sorted(
        (split, cls)
        for split in splits
        for cls in required
        if not any(
            r["split"] == split and r["class"] == cls
            for r in accepted
        )
    )

    if missing:
        print("MISSING_CLASS_SPLIT:", missing)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())