"""Phase 42.1: deterministic annotated-video *prediction* evaluator.

This module never opens a camera, executes an AI model, or estimates GPS/range.
It evaluates a supplied JSONL of per-frame ground truth and model predictions.

Geometry matching is class-agnostic and globally one-to-one per frame; a
classification error is therefore visible instead of being silently dropped.
Drone identity switches are measured ONLY against supplied ground-truth IDs.
The resulting metrics are not a claim of field performance until measured on
licensed, independent, representative annotated footage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

VERSION = "skyguard.phase42.benchmark.v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_LABEL = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
_MAX_OBJECTS_PER_FRAME = 128
_MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024


class BenchmarkInputError(ValueError):
    """Invalid annotated-video benchmark input; no metrics are produced."""


def _object(value: Any, name: str, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkInputError(f"{name} must be an object")
    unknown = set(value) - allowed
    missing = required - set(value)
    if missing or unknown:
        raise BenchmarkInputError(f"{name} keys: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise BenchmarkInputError(f"{name} must be an integer in [{low},{high}]")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise BenchmarkInputError(f"{name} must be a finite number in [{low},{high}]")
    return float(value)


def _identifier(value: Any, name: str) -> str:
    # Keep strings and integers in disjoint namespaces, so id 7 != id "7".
    if isinstance(value, str) and _ID.fullmatch(value):
        return "s:" + value
    if type(value) is int and 0 <= value <= 2**53 - 1:
        return "i:" + str(value)
    raise BenchmarkInputError(f"{name} must be a safe nonempty ID string or nonnegative integer")


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        raise BenchmarkInputError(f"{name} must be a lowercase class label")
    return value


def _bbox(value: Any, name: str, width: int, height: int) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise BenchmarkInputError(f"{name} must contain four [x1,y1,x2,y2] pixel values")
    x1 = _number(value[0], name + ".x1", 0, width)
    y1 = _number(value[1], name + ".y1", 0, height)
    x2 = _number(value[2], name + ".x2", 0, width)
    y2 = _number(value[3], name + ".y2", 0, height)
    if x2 <= x1 or y2 <= y1:
        raise BenchmarkInputError(f"{name} must have strictly positive area")
    return x1, y1, x2, y2


def _parse_frame(raw: Any) -> dict[str, Any]:
    frame = _object(
        raw, "frame",
        {"sequence_id", "frame_index", "width", "height", "ground_truth", "predictions"},
        {"sequence_id", "frame_index", "width", "height", "ground_truth", "predictions"},
    )
    seq = _identifier(frame["sequence_id"], "sequence_id")
    idx = _integer(frame["frame_index"], "frame_index", 0, 10**12)
    width = _integer(frame["width"], "width", 1, 32768)
    height = _integer(frame["height"], "height", 1, 32768)
    truth = frame["ground_truth"]
    pred = frame["predictions"]
    for key, items in (("ground_truth", truth), ("predictions", pred)):
        if not isinstance(items, list) or len(items) > _MAX_OBJECTS_PER_FRAME:
            raise BenchmarkInputError(f"{key} must be a list of at most {_MAX_OBJECTS_PER_FRAME} objects")
    gts: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    seen_gt: set[str] = set()
    seen_pred: set[str] = set()
    for n, data in enumerate(truth):
        item = _object(data, f"ground_truth[{n}]", {"id", "class", "bbox_xyxy"}, {"id", "class", "bbox_xyxy"})
        gid = _identifier(item["id"], "ground_truth.id")
        if gid in seen_gt:
            raise BenchmarkInputError("duplicate ground-truth ID in one frame")
        seen_gt.add(gid)
        gts.append({"id": gid, "class": _label(item["class"], "ground_truth.class"),
                    "bbox": _bbox(item["bbox_xyxy"], "ground_truth.bbox_xyxy", width, height)})
    for n, data in enumerate(pred):
        item = _object(data, f"predictions[{n}]", {"track_id", "class", "bbox_xyxy", "confidence"},
                       {"class", "bbox_xyxy", "confidence"})
        tid = None if "track_id" not in item or item["track_id"] is None else _identifier(item["track_id"], "predictions.track_id")
        if tid is not None and tid in seen_pred:
            raise BenchmarkInputError("duplicate predicted track_id in one frame")
        if tid is not None:
            seen_pred.add(tid)
        preds.append({"track_id": tid, "class": _label(item["class"], "predictions.class"),
                      "confidence": _number(item["confidence"], "predictions.confidence", 0, 1),
                      "bbox": _bbox(item["bbox_xyxy"], "predictions.bbox_xyxy", width, height)})
    return {"sequence_id": seq, "frame_index": idx, "ground_truth": gts, "predictions": preds}


def bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    """Return intersection-over-union of two validated xyxy boxes."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    a = (left[2] - left[0]) * (left[3] - left[1])
    b = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (a + b - intersection)


def _optimal_match(gt: list[dict[str, Any]], pred: list[dict[str, Any]], threshold: float) -> list[tuple[int, int, float]]:
    """Hungarian maximum-weight one-to-one assignment, then IoU threshold.

    A cardinality bonus ensures matching as many eligible pairs as possible
    before maximizing their total IoU. Ties are deterministic by input order.
    """
    if not gt or not pred:
        return []
    size = max(len(gt), len(pred))
    bonus = size + 1
    overlaps = [[bbox_iou(g["bbox"], p["bbox"]) for p in pred] for g in gt]
    weights = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(len(gt)):
        for j in range(len(pred)):
            if overlaps[i][j] >= threshold:
                weights[i][j] = bonus + overlaps[i][j]

    # O(N^3) Hungarian algorithm for square cost matrix, 1-indexed internals.
    max_weight = bonus + 1.0
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = (max_weight - weights[i0 - 1][j - 1]) - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out: list[tuple[int, int, float]] = []
    for col in range(1, size + 1):
        row = p[col] - 1
        j = col - 1
        if row < len(gt) and j < len(pred) and overlaps[row][j] >= threshold:
            out.append((row, j, overlaps[row][j]))
    return sorted(out)


def _score(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_frames(frames: Iterable[Mapping[str, Any]], *, iou_threshold: float = 0.5,
                    confidence_threshold: float = 0.25, max_switch_gap: int = 30) -> dict[str, Any]:
    """Evaluate annotated frames. Predicted IDs are never treated as ground truth."""
    iou_threshold = _number(iou_threshold, "iou_threshold", 0.0000001, 1)
    confidence_threshold = _number(confidence_threshold, "confidence_threshold", 0, 1)
    max_switch_gap = _integer(max_switch_gap, "max_switch_gap", 1, 10**6)
    per_cls: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    seen_last_index: dict[str, int] = {}
    prev_association: dict[tuple[str, str], tuple[str, int]] = {}
    predicted_drone_ids: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    sequence_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_iou = 0.0
    for position, raw in enumerate(frames, start=1):
        try:
            frame = _parse_frame(raw)
            seq = frame["sequence_id"]
            idx = frame["frame_index"]
            if seq in seen_last_index and idx <= seen_last_index[seq]:
                raise BenchmarkInputError("frame_index must increase strictly per sequence")
            seen_last_index[seq] = idx
            gt = frame["ground_truth"]
            pred_all = frame["predictions"]
            pred = [p for p in pred_all if p["confidence"] >= confidence_threshold]
            dropped = len(pred_all) - len(pred)
            counts["frames"] += 1
            counts["gt_objects"] += len(gt)
            counts["pred_objects"] += len(pred)
            counts["predictions_below_threshold"] += dropped
            seq_counts = sequence_counts[seq]
            seq_counts["frames"] += 1
            seq_counts["gt_objects"] += len(gt)
            seq_counts["pred_objects"] += len(pred)
            gt_drones = sum(g["class"] == "drone" for g in gt)
            pr_drones = sum(p["class"] == "drone" for p in pred)
            counts["max_simultaneous_gt_drones"] = max(counts["max_simultaneous_gt_drones"], gt_drones)
            counts["max_simultaneous_predicted_drones"] = max(counts["max_simultaneous_predicted_drones"], pr_drones)
            seq_counts["max_simultaneous_gt_drones"] = max(seq_counts["max_simultaneous_gt_drones"], gt_drones)
            seq_counts["max_simultaneous_predicted_drones"] = max(seq_counts["max_simultaneous_predicted_drones"], pr_drones)
            for p in pred:
                if p["class"] == "drone" and p["track_id"] is not None:
                    predicted_drone_ids.add((seq, p["track_id"]))
            matched = _optimal_match(gt, pred, iou_threshold)
            gi = {a for a, _, _ in matched}
            pi = {b for _, b, _ in matched}
            counts["localization_matches"] += len(matched)
            for i, j, overlap in matched:
                g, p = gt[i], pred[j]
                total_iou += overlap
                confusion[g["class"]][p["class"]] += 1
                if g["class"] == p["class"]:
                    per_cls[g["class"]]["tp"] += 1
                    counts["tp"] += 1
                    seq_counts["tp"] += 1
                else:
                    per_cls[g["class"]]["fn"] += 1
                    per_cls[p["class"]]["fp"] += 1
                    counts["fp"] += 1
                    counts["fn"] += 1
                    seq_counts["fp"] += 1
                    seq_counts["fn"] += 1
                if g["class"] == "drone" and p["class"] == "drone":
                    counts["matched_drone_observations"] += 1
                    seq_counts["matched_drone_observations"] += 1
                    tid = p["track_id"]
                    if tid is None:
                        counts["matched_drones_without_track_id"] += 1
                    else:
                        key = (seq, g["id"])
                        previous = prev_association.get(key)
                        if previous is not None and 0 < idx - previous[1] <= max_switch_gap and previous[0] != tid:
                            counts["id_switches"] += 1
                            seq_counts["id_switches"] += 1
                        prev_association[key] = (tid, idx)
            for i, g in enumerate(gt):
                if i not in gi:
                    confusion[g["class"]]["__missed__"] += 1
                    per_cls[g["class"]]["fn"] += 1
                    counts["fn"] += 1
                    seq_counts["fn"] += 1
            for j, p in enumerate(pred):
                if j not in pi:
                    confusion["__spurious__"][p["class"]] += 1
                    per_cls[p["class"]]["fp"] += 1
                    counts["fp"] += 1
                    seq_counts["fp"] += 1
        except BenchmarkInputError as exc:
            raise BenchmarkInputError(f"frame record {position}: {exc}") from exc
    if not counts["frames"]:
        raise BenchmarkInputError("benchmark contains no frames")
    classes = sorted(per_cls)
    return {
        "schema_version": VERSION,
        "evidence_type": "offline_annotated_video_predictions_unverified_provenance",
        "not_measured": ["physical_drone_identity", "range_m", "target_gps", "camera_fps", "inference_latency", "deployment_cost"],
        "configuration": {"iou_threshold": iou_threshold, "confidence_threshold": confidence_threshold,
                          "max_switch_gap_frames": max_switch_gap, "match_policy": "class_agnostic_max_cardinality_then_total_iou"},
        "frames": counts["frames"],
        "sequences": len(seen_last_index),
        "ground_truth_objects": counts["gt_objects"],
        "retained_predictions": counts["pred_objects"],
        "predictions_below_threshold": counts["predictions_below_threshold"],
        "detection_classification": _score(counts["tp"], counts["fp"], counts["fn"]),
        "per_class": {name: _score(per_cls[name]["tp"], per_cls[name]["fp"], per_cls[name]["fn"]) for name in classes},
        "localization": {"matched_pairs": counts["localization_matches"],
                         "mean_matched_iou": total_iou / counts["localization_matches"] if counts["localization_matches"] else None},
        "confusion": {name: dict(sorted(items.items())) for name, items in sorted(confusion.items())},
        "drone_tracking": {
            "id_switches": counts["id_switches"],
            "matched_drone_observations": counts["matched_drone_observations"],
            "matched_drones_without_track_id": counts["matched_drones_without_track_id"],
            "distinct_predicted_drone_track_ids": len(predicted_drone_ids),
            "max_simultaneous_ground_truth_drones": counts["max_simultaneous_gt_drones"],
            "max_simultaneous_predicted_drones": counts["max_simultaneous_predicted_drones"],
            "note": "ID switches compare associations of labeled GT drone IDs and predicted drone track IDs within the configured frame gap. Not an IDF1/MOTA benchmark."
        },
        "per_sequence": {
            seq: {
                "frames": c["frames"], "detection_classification": _score(c["tp"], c["fp"], c["fn"]),
                "id_switches": c["id_switches"],
                "max_simultaneous_gt_drones": c["max_simultaneous_gt_drones"],
                "max_simultaneous_predicted_drones": c["max_simultaneous_predicted_drones"]
            } for seq, c in sorted(sequence_counts.items())
        },
    }


def read_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    input_path = Path(path)
    with input_path.open("rb") as file:
        for line_no, raw in enumerate(file, 1):
            if len(raw) > _MAX_JSONL_LINE_BYTES:
                raise BenchmarkInputError(f"line {line_no} exceeds maximum JSONL line size")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw.decode("utf-8"), parse_constant=lambda const: (_ for _ in ()).throw(ValueError("invalid constant " + const)))
            except (UnicodeError, ValueError) as exc:
                raise BenchmarkInputError(f"invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise BenchmarkInputError(f"line {line_no} must contain a frame object")
            yield obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SkyGuard 42.1 deterministic offline annotated prediction evaluator (not physical range/GPS)")
    parser.add_argument("--input", required=True, help="Ground-truth + prediction JSONL")
    parser.add_argument("--output", help="JSON results; omitted means stdout (refuses to overwrite existing files)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--max-switch-gap", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        if args.output and Path(args.output).exists():
            raise BenchmarkInputError("output already exists; choose a new report path")
        report = evaluate_frames(read_jsonl(args.input), iou_threshold=args.iou_threshold,
                                 confidence_threshold=args.confidence_threshold, max_switch_gap=args.max_switch_gap)
        rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            # Exclusive creation: no accidental overwrite of a previous benchmark.
            with Path(args.output).open("x", encoding="utf-8") as out:
                out.write(rendered)
            print("PHASE_42_1_OFFLINE_BENCHMARK_REPORT_WRITTEN: " + args.output)
        else:
            print(rendered, end="")
        return 0
    except (BenchmarkInputError, OSError) as exc:
        print("PHASE_42_1_BENCHMARK_REJECTED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
