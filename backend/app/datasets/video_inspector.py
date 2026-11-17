from dataclasses import dataclass
from pathlib import Path
import math

import cv2

from app.datasets.manifest import VideoManifest


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    first_frame_decodable: bool


def inspect_video(path: Path) -> VideoMetadata:
    if not path.is_file():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot open video")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        ok, frame = capture.read()

        if not ok or frame is None:
            raise ValueError("Cannot decode first frame")

        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions")

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video FPS")

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=count,
            first_frame_decodable=True,
        )
    finally:
        capture.release()


def validate_video_metadata(
    manifest: VideoManifest,
    path: Path,
    fps_tolerance: float = 0.5,
) -> VideoMetadata:
    metadata = inspect_video(path)

    if metadata.width != manifest.width:
        raise ValueError("Width mismatch")

    if metadata.height != manifest.height:
        raise ValueError("Height mismatch")

    if abs(metadata.fps - manifest.fps) > fps_tolerance:
        raise ValueError("FPS mismatch")

    return metadata
