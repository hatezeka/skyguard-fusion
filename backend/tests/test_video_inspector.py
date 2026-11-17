import cv2
import numpy as np
import pytest

from app.datasets.video_inspector import (
    inspect_video,
    validate_video_metadata,
)
from app.datasets.manifest import VideoManifest


def make_video(path, width=320, height=240, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(path), fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")

    for index in range(10):
        frame = np.zeros(
            (height, width, 3), dtype=np.uint8
        )
        cv2.circle(
            frame,
            (30 + index * 10, 100),
            8,
            (255, 255, 255),
            -1,
        )
        writer.write(frame)

    writer.release()


def make_manifest(width=320, height=240, fps=10):
    return VideoManifest(
        video_id="TEST-001",
        file_path="test.avi",
        source_type="synthetic",
        width=width,
        height=height,
        fps=fps,
        split="test",
        scenario_tags=["drone"],
        license="synthetic-test",
    )


def test_inspect_video(tmp_path):
    path = tmp_path / "test.avi"
    make_video(path)

    result = inspect_video(path)

    assert result.width == 320
    assert result.height == 240
    assert result.frame_count == 10
    assert result.first_frame_decodable


def test_validate_video_metadata(tmp_path):
    path = tmp_path / "test.avi"
    make_video(path)

    result = validate_video_metadata(
        make_manifest(), path
    )

    assert result.fps == pytest.approx(10, abs=0.5)


def test_width_mismatch(tmp_path):
    path = tmp_path / "test.avi"
    make_video(path)

    with pytest.raises(ValueError, match="Width mismatch"):
        validate_video_metadata(
            make_manifest(width=640), path
        )


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_video(tmp_path / "missing.avi")


def test_corrupt_video(tmp_path):
    path = tmp_path / "broken.avi"
    path.write_bytes(b"not a video")

    with pytest.raises(ValueError):
        inspect_video(path)
