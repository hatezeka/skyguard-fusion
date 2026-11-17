import json
import math
from collections import Counter
from pathlib import Path


def validate_annotations(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))

    width = payload["width"]
    height = payload["height"]
    expected_frames = payload["frame_count"]
    annotations = payload["annotations"]

    if not isinstance(width, int) or width <= 0:
        raise ValueError("Invalid width")

    if not isinstance(height, int) or height <= 0:
        raise ValueError("Invalid height")

    if not isinstance(expected_frames, int) or expected_frames <= 0:
        raise ValueError("Invalid frame count")

    if len(annotations) != expected_frames:
        raise ValueError("Frame count mismatch")

    visible_count = 0
    hidden_count = 0
    track_ids = set()

    for expected_index, frame in enumerate(annotations):
        if frame["frame_index"] != expected_index:
            raise ValueError("Invalid frame sequence")

        ids_in_frame = set()

        for obj in frame["objects"]:
            track_id = obj["track_id"]

            if not isinstance(track_id, int) or track_id <= 0:
                raise ValueError("Invalid track ID")

            if track_id in ids_in_frame:
                raise ValueError("Duplicate track ID in frame")

            ids_in_frame.add(track_id)
            track_ids.add(track_id)

            x, y = obj["center"]
            bx, by, bw, bh = obj["bbox"]

            if not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in (x, y, bx, by, bw, bh)
            ):
                raise ValueError("Non-finite coordinates")

            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("Center outside frame")

            if bw <= 0 or bh <= 0:
                raise ValueError("Invalid bounding box")

            if not (0 <= bx and 0 <= by):
                raise ValueError("Bounding box outside frame")

            if bx + bw > width or by + bh > height:
                raise ValueError("Bounding box outside frame")

            if not isinstance(obj["visible"], bool):
                raise ValueError("Invalid visibility")

            if obj["visible"]:
                visible_count += 1
            else:
                hidden_count += 1

    return {
        "valid": True,
        "frame_count": expected_frames,
        "unique_tracks": len(track_ids),
        "visible_instances": visible_count,
        "hidden_instances": hidden_count,
    }


def validate_split_integrity(records: list[dict]) -> dict:
    source_splits = {}

    for record in records:
        source = record["source_id"]
        split = record["split"]

        if not isinstance(source, str) or not source:
            raise ValueError("Invalid source ID")

        if split not in {"train", "validation", "test"}:
            raise ValueError("Invalid dataset split")

        if source in source_splits and source_splits[source] != split:
            raise ValueError("Dataset source leakage")

        source_splits[source] = split

    return {
        "valid": True,
        "sources": len(source_splits),
        "split_counts": dict(Counter(source_splits.values())),
    }
