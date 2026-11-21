"""Phase 42.2: turn LOCAL video + independent GT + real YOLO checkpoint
into Phase 42.1's benchmark JSONL. Never infers target depth, range, or GPS.

This baseline invokes Ultralytics YOLO.track (ByteTrack by default), NOT the
SkyGuard LiveSessionManager tracker. Do not present baseline MOT results as
production tracker performance. All video frames are processed in order;
only annotated frames are scored; missing GT frames require an explicit flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
from typing import Any, Mapping

from .mot_benchmark import BenchmarkInputError, _parse_frame, evaluate_frames, read_jsonl

VERSION = "skyguard.phase42.real_video_adapter.v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
LABEL = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MAX_JSONL_BYTES = 4 * 1024 * 1024


class RealVideoError(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def class_map_from_json(path: Path) -> dict[str, str]:
    """Explicit mapping, with no silent class reinterpretation."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RealVideoError(f"Cannot read class map: {exc}") from exc
    if not isinstance(raw, dict) or not raw or len(raw) > 100:
        raise RealVideoError("class map must be nonempty object of at most 100 entries")
    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key or len(key) > 100:
            raise RealVideoError("invalid model class name in class map")
        if not isinstance(value, str) or not LABEL.fullmatch(value):
            raise RealVideoError("class map labels must be lowercase safe class names")
        mapping[key] = value
    return mapping


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        entries = names.items()
    elif isinstance(names, (list, tuple)):
        entries = enumerate(names)
    else:
        raise RealVideoError("YOLO model.names must be index/name mapping")
    result: dict[int, str] = {}
    for cls_id, name in entries:
        if type(cls_id) is not int or cls_id < 0 or not isinstance(name, str) or not name:
            raise RealVideoError("invalid YOLO model class metadata")
        result[cls_id] = name
    if not result:
        raise RealVideoError("YOLO has no class names")
    return result


def validate_class_map(names: dict[int, str], mapping: dict[str, str]) -> None:
    missing = set(names.values()) - set(mapping)
    extra = set(mapping) - set(names.values())
    if missing or extra:
        raise RealVideoError(f"class map mismatch: unmapped={sorted(missing)}, unknown={sorted(extra)}")


def read_independent_annotations(path: Path, sequence_id: str) -> dict[int, dict[str, Any]]:
    """Independent GT ONLY. Reject any predictions in the annotation file."""
    if not SAFE_ID.fullmatch(sequence_id):
        raise RealVideoError("sequence_id must be a safe identifier")
    if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        raise RealVideoError("GT JSONL missing or exceeds 256 MiB")
    annotations: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line_no, blob in enumerate(handle, 1):
            if not blob.strip():
                continue
            if len(blob) > MAX_JSONL_BYTES:
                raise RealVideoError(f"GT line {line_no} too large")
            try:
                item = json.loads(blob.decode("utf-8"), parse_constant=lambda n: (_ for _ in ()).throw(ValueError(n)))
            except (UnicodeError, ValueError) as exc:
                raise RealVideoError(f"Invalid JSON at GT line {line_no}: {exc}") from exc
            if not isinstance(item, dict) or set(item) != {"sequence_id", "frame_index", "width", "height", "ground_truth"}:
                raise RealVideoError(f"GT line {line_no} must have exactly GT-only keys (no predictions)")
            if item["sequence_id"] != sequence_id:
                raise RealVideoError(f"GT line {line_no} sequence_id mismatch")
            # Reuse exact strict 42.1 input validation, adding empty predictions only internally.
            validated = _parse_frame({**item, "predictions": []})
            index = validated["frame_index"]
            if index in annotations:
                raise RealVideoError(f"GT frame {index} duplicated")
            annotations[index] = item
            if len(annotations) > 200_000:
                raise RealVideoError("too many GT records")
    if not annotations:
        raise RealVideoError("no independent ground-truth annotations")
    return dict(sorted(annotations.items()))


def _tolist(field: Any, label: str) -> list[Any]:
    if field is None:
        raise RealVideoError(f"YOLO result missing {label}")
    raw = field.cpu().tolist() if hasattr(field, "cpu") else field.tolist() if hasattr(field, "tolist") else field
    if not isinstance(raw, (list, tuple)):
        raise RealVideoError(f"YOLO {label} is not an array")
    return list(raw)


def parse_yolo_predictions(result: Any, names: dict[int, str], mapping: dict[str, str], width: int, height: int) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    coords = _tolist(getattr(boxes, "xyxy", None), "boxes.xyxy")
    confs = _tolist(getattr(boxes, "conf", None), "boxes.conf")
    classes = _tolist(getattr(boxes, "cls", None), "boxes.cls")
    ids_obj = getattr(boxes, "id", None)
    ids = [None] * len(coords) if ids_obj is None else _tolist(ids_obj, "boxes.id")
    if not len(coords) == len(confs) == len(classes) == len(ids):
        raise RealVideoError("YOLO boxes field lengths disagree")
    if len(coords) > 128:
        raise RealVideoError("more than 128 detections in one frame: exceeds benchmark input contract")
    output: list[dict[str, Any]] = []
    assigned: set[int] = set()
    for xyxy, confidence, class_id, track_id in zip(coords, confs, classes, ids):
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            raise RealVideoError("YOLO bounding box must be xyxy of four values")
        if type(class_id) not in (int, float) or not math.isfinite(class_id) or int(class_id) != class_id or int(class_id) not in names:
            raise RealVideoError(f"YOLO reported unmapped class id {class_id!r}")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise RealVideoError("YOLO confidence must be finite within [0,1]")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in xyxy):
            raise RealVideoError("YOLO bounding box contains nonfinite/non-numeric values")
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise RealVideoError("YOLO bounding box zero area after clipping to source frame")
        row: dict[str, Any] = {"class": mapping[names[int(class_id)]],
                               "bbox_xyxy": [x1, y1, x2, y2],
                               "confidence": float(confidence)}
        if track_id is not None:
            if type(track_id) not in (int, float) or not math.isfinite(track_id) or int(track_id) != track_id or track_id < 0 or track_id > 2**53 - 1:
                raise RealVideoError("invalid YOLO track_id")
            key = int(track_id)
            if key in assigned:
                raise RealVideoError("YOLO reused one track ID twice within a frame")
            assigned.add(key)
            row["track_id"] = key
        output.append(row)
    return output


def merge_frame(ground_truth: Mapping[str, Any], predictions: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    if ground_truth["width"] != width or ground_truth["height"] != height:
        raise RealVideoError("Video frame resolution differs from independent GT annotation")
    record = {**ground_truth, "predictions": predictions}
    _parse_frame(record)
    return record


def process_frames(capture: Any, model: Any, annotations: dict[int, dict[str, Any]],
                   names: dict[int, str], mapping: dict[str, str], *, max_frames: int,
                   confidence: float, imgsz: int, device: str, tracker: str,
                   allow_partial: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every decoded frame passes through track, even when it lacks GT."""
    rows: list[dict[str, Any]] = []
    decoded = 0
    total_seconds = 0.0
    wall_started = time.perf_counter()
    latencies: list[float] = []
    first_width = first_height = None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if decoded >= max_frames:
            raise RealVideoError(f"Video exceeds --max-frames {max_frames}; select a bounded clip")
        height, width = image.shape[:2]
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise RealVideoError("invalid decoded frame dimensions")
        if first_width is None:
            first_width, first_height = width, height
        if (width, height) != (first_width, first_height):
            raise RealVideoError("video resolution changed mid-sequence")
        start = time.perf_counter()
        results = model.track(image, persist=True, verbose=False, conf=confidence,
                              imgsz=imgsz, device=device, tracker=tracker)
        latency = time.perf_counter() - start
        total_seconds += latency
        latencies.append(latency)
        if not isinstance(results, (list, tuple)) or len(results) != 1:
            raise RealVideoError("YOLO.track did not produce exactly one result per decoded frame")
        preds = parse_yolo_predictions(results[0], names, mapping, width, height)
        if decoded in annotations:
            rows.append(merge_frame(annotations[decoded], preds, width, height))
        elif not allow_partial:
            raise RealVideoError(f"Video frame {decoded} has no independent GT record; annotate all frames or use --allow-partial")
        decoded += 1
    if decoded == 0:
        raise RealVideoError("No frames decoded; unsupported or empty video")
    if max(annotations) >= decoded:
        raise RealVideoError("GT references frames beyond decoded video")
    if len(rows) != len(annotations):
        raise RealVideoError("not all GT annotations were matched to video frames")
    elapsed_wall = time.perf_counter() - wall_started
    ordered_latency = sorted(latencies)
    return rows, {"decoded_frames": decoded, "annotated_frames": len(rows),
                  "annotation_coverage": len(rows) / decoded,
                  "model_inference_seconds_total": total_seconds,
                  "model_latency_p50_ms": 1000 * ordered_latency[(len(ordered_latency) - 1) // 2],
                  "model_latency_p95_ms": 1000 * ordered_latency[math.ceil(.95 * len(ordered_latency)) - 1],
                  "wall_seconds_total_including_decode": elapsed_wall,
                  "wall_fps_including_decode": decoded / elapsed_wall if elapsed_wall > 0 else None,
                  "model_inference_fps_mean": (decoded / total_seconds) if total_seconds > 0 else None,
                  "source_frame_width": first_width, "source_frame_height": first_height,
                  "partial_annotation_coverage": len(rows) != decoded}


def run_real_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    video = Path(args.video).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    gt_path = Path(args.ground_truth).expanduser().resolve()
    map_path = Path(args.class_map).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    for file, name in ((video, "video"), (weights, "checkpoint"), (gt_path, "independent GT"), (map_path, "class map")):
        if not file.is_file():
            raise RealVideoError(f"{name} does not exist: {file}")
    if video.suffix.lower() not in VIDEO_EXT:
        raise RealVideoError("unsupported video extension")
    if weights.suffix.lower() != ".pt":
        raise RealVideoError("Ultralytics checkpoint must be a local .pt file")
    if out_dir.exists():
        raise RealVideoError("Output directory exists. Choose a new path to protect earlier reports")
    if args.max_frames < 1 or args.max_frames > 200000:
        raise RealVideoError("max_frames must be between 1 and 200000")
    if not 0 < args.confidence <= 1 or not math.isfinite(args.confidence):
        raise RealVideoError("confidence must be in (0,1]")
    if args.imgsz < 64 or args.imgsz > 4096:
        raise RealVideoError("imgsz out of supported range")
    if args.tracker not in ("bytetrack.yaml", "botsort.yaml"):
        raise RealVideoError("tracker must be bytetrack.yaml or botsort.yaml")
    if not 0 < args.iou_threshold <= 1 or not math.isfinite(args.iou_threshold):
        raise RealVideoError("IoU threshold must be finite in (0,1]")
    if args.max_switch_gap < 1 or args.max_switch_gap > 1_000_000:
        raise RealVideoError("max_switch_gap outside supported range")
    if not args.confirm_license:
        raise RealVideoError("Set --confirm-license after reviewing rights for local video and annotations")
    if not args.confirm_trusted_checkpoint:
        raise RealVideoError("Set --confirm-trusted-checkpoint only for a known trusted local PyTorch .pt checkpoint")
    annotations = read_independent_annotations(gt_path, args.sequence_id)
    mapping = class_map_from_json(map_path)
    # Ultralytics must not automatically install missing runtime packages.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        import cv2
        from ultralytics import YOLO
        import ultralytics
    except ImportError as exc:
        raise RealVideoError(f"Missing optional real-inference dependency: {exc}. No packages installed") from exc
    model = YOLO(str(weights))
    names = normalize_names(model.names)
    validate_class_map(names, mapping)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RealVideoError("OpenCV failed to open local video")
    try:
        rows, performance = process_frames(capture, model, annotations, names, mapping,
                   max_frames=args.max_frames, confidence=args.confidence,
                   imgsz=args.imgsz, device=args.device, tracker=args.tracker,
                   allow_partial=args.allow_partial)
    finally:
        capture.release()
    metrics = evaluate_frames(rows, confidence_threshold=args.confidence,
                              iou_threshold=args.iou_threshold,
                              max_switch_gap=args.max_switch_gap)
    run_info = {"schema_version": VERSION, "baseline": "ultralytics_yolo_track_not_skyguard_live_tracker",
                "video_sha256": digest(video), "checkpoint_sha256": digest(weights),
                "independent_gt_sha256": digest(gt_path), "class_map_sha256": digest(map_path),
                "sequence_id": args.sequence_id, "tracker": args.tracker, "device": args.device,
                "model_names": names, "class_map": mapping, "model_confidence_threshold": args.confidence,
                "evaluation_iou_threshold": args.iou_threshold, "max_switch_gap": args.max_switch_gap,
                "image_size": args.imgsz, "python_version": platform.python_version(),
                "opencv_version": cv2.__version__, "ultralytics_version": ultralytics.__version__,
                "performance": performance,
                "limitations": ["Metrics depend on independent frame annotations and ground truth quality",
                    "This is Ultralytics tracking baseline, NOT SkyGuard production tracker",
                    "Missing annotations excluded only if explicitly allowed",
                    "FPS refers to model.track call time; excludes video decoding and UI",
                    "No target GPS, physical range, or facility distance measured"]}
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        with (out_dir / "annotated_predictions.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (out_dir / "manifest.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return {"output_dir": str(out_dir), "frames_scored": metrics["frames"], "decoded_frames": performance["decoded_frames"],
            "drone_id_switches": metrics["drone_tracking"]["id_switches"],
            "note": "Actual local video inference + independent GT; not SkyGuard live tracker or physical GPS/range"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SkyGuard 42.2 local REAL YOLO baseline with independent annotated video; NO GPS/range")
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--ground-truth", required=True, help="GT-only JSONL, one record per annotated frame")
    parser.add_argument("--class-map", required=True, help="JSON mapping model class names to benchmark labels")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--output-dir", required=True, help="must not already exist")
    parser.add_argument("--confirm-license", action="store_true")
    parser.add_argument("--confirm-trusted-checkpoint", action="store_true", help=".pt checkpoint must come from a trusted source")
    parser.add_argument("--allow-partial", action="store_true", help="explicitly exclude unannotated frames from metrics")
    parser.add_argument("--max-frames", type=int, default=3000)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-switch-gap", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 1 etc as supported by Ultralytics")
    parser.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    args = parser.parse_args(argv)
    try:
        result = run_real_benchmark(args)
    except (RealVideoError, BenchmarkInputError, OSError, ValueError) as exc:
        print("PHASE_42_2_REJECTED: " + str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print("PHASE_42_2_INFERENCE_FAILED: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 3
    print("PHASE_42_2_REAL_VIDEO_BENCHMARK_WRITTEN: " + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
