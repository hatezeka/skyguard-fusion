from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

import cv2


CATEGORIES = {"drone", "bird", "negative"}


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    category: str
    source_url: str
    license_name: str
    license_verified: bool
    sha256: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    annotation_status: str = "unannotated"
    split: str = "unassigned"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(
    path: Path,
    category: str,
    source_url: str,
    license_name: str,
    license_verified: bool = False,
) -> VideoRecord:
    if category not in CATEGORIES:
        raise ValueError("Unsupported category")

    if not path.is_file():
        raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if (
            width <= 0
            or height <= 0
            or not math.isfinite(fps)
            or fps <= 0
            or frames <= 0
        ):
            raise ValueError("Invalid video metadata")

        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("Video has no readable frames")
    finally:
        capture.release()

    checksum = sha256_file(path)

    return VideoRecord(
        video_id=checksum[:16],
        category=category,
        source_url=source_url,
        license_name=license_name,
        license_verified=license_verified,
        sha256=checksum,
        width=width,
        height=height,
        fps=fps,
        frame_count=frames,
        duration_seconds=round(frames / fps, 3),
    )


def save_manifest(record: VideoRecord, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(record), stream, indent=2)
        stream.write("\n")