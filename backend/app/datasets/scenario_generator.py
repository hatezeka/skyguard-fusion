import json
from pathlib import Path

import cv2
import numpy as np


def generate_scenario(
    output_dir: Path,
    scenario: str,
    width: int = 640,
    height: int = 360,
    fps: int = 20,
    frames: int = 100,
):
    allowed = {"crossing", "occlusion", "multiple_drones"}

    if scenario not in allowed:
        raise ValueError("Unknown scenario")

    if width < 64 or height < 64 or fps <= 0 or frames <= 0:
        raise ValueError("Invalid video parameters")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / f"{scenario}.avi"
    annotations_path = output_dir / f"{scenario}.json"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(video_path), fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Cannot initialize video writer")

    annotations = []

    try:
        for frame_index in range(frames):
            frame = np.zeros(
                (height, width, 3), dtype=np.uint8
            )

            objects = []

            if scenario == "crossing":
                positions = [
                    (1, int(width * (0.1 + 0.8 * frame_index / frames)),
                     int(height * 0.45)),
                    (2, int(width * (0.9 - 0.8 * frame_index / frames)),
                     int(height * 0.55)),
                ]

            elif scenario == "occlusion":
                positions = [
                    (1, int(width * (0.1 + 0.8 * frame_index / frames)),
                     int(height * 0.5)),
                ]

            else:
                positions = [
                    (
                        object_id,
                        int(width * (0.1 + 0.7 * frame_index / frames)),
                        int(height * (0.2 + object_id * 0.15)),
                    )
                    for object_id in range(1, 5)
                ]

            for object_id, x, y in positions:
                radius = 7
                visible = True

                if scenario == "occlusion":
                    visible = not (
                        width * 0.43 <= x <= width * 0.57
                    )

                if visible:
                    cv2.circle(
                        frame, (x, y), radius,
                        (255, 255, 255), -1
                    )

                objects.append({
                    "track_id": object_id,
                    "class": "synthetic_target",
                    "center": [x, y],
                    "bbox": [
                        x - radius, y - radius,
                        radius * 2, radius * 2
                    ],
                    "visible": visible,
                })

            if scenario == "occlusion":
                left = int(width * 0.43)
                right = int(width * 0.57)
                cv2.rectangle(
                    frame,
                    (left, 0),
                    (right, height - 1),
                    (60, 60, 60),
                    -1,
                )

            writer.write(frame)

            annotations.append({
                "frame_index": frame_index,
                "objects": objects,
            })
    finally:
        writer.release()

    payload = {
        "scenario": scenario,
        "source_type": "synthetic",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frames,
        "annotations": annotations,
    }

    annotations_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return video_path, annotations_path
