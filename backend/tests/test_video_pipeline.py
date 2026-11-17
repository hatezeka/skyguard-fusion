import cv2
import numpy as np
import pytest

from app.video.pipeline import VideoProcessingPipeline
from app.video.source import VideoSourceConfig


def make_video(path, count=8):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (160, 120),
    )

    assert writer.isOpened()

    try:
        for index in range(count):
            image = np.full(
                (120, 160, 3),
                index * 20,
                dtype=np.uint8,
            )
            writer.write(image)
    finally:
        writer.release()


def test_pipeline_processes_all_frames(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path)

    collected = []

    pipeline = VideoProcessingPipeline(
        VideoSourceConfig("file", str(path))
    )

    result = pipeline.run(on_frame=collected.append)

    assert result.frames_processed == 8
    assert len(collected) == 8
    assert result.processing_fps > 0


def test_pipeline_respects_frame_limit(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path)

    pipeline = VideoProcessingPipeline(
        VideoSourceConfig("file", str(path))
    )

    result = pipeline.run(max_frames=3)

    assert result.frames_processed == 3


def test_pipeline_applies_camera_profile(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path)

    collected = []

    pipeline = VideoProcessingPipeline(
        VideoSourceConfig("file", str(path)),
        profile="low_light",
    )

    pipeline.run(on_frame=collected.append)

    assert len(collected) == 8
    assert all(
        frame.profile == "low_light"
        for frame in collected
    )


def test_pipeline_preserves_frame_indices(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path)

    collected = []

    pipeline = VideoProcessingPipeline(
        VideoSourceConfig("file", str(path))
    )

    pipeline.run(on_frame=collected.append)

    assert [frame.index for frame in collected] == list(range(8))


def test_pipeline_rejects_invalid_frame_limit(tmp_path):
    path = tmp_path / "video.avi"
    make_video(path)

    pipeline = VideoProcessingPipeline(
        VideoSourceConfig("file", str(path))
    )

    with pytest.raises(ValueError, match="max_frames"):
        pipeline.run(max_frames=0)


def test_pipeline_rejects_missing_video(tmp_path):
    pipeline = VideoProcessingPipeline(
        VideoSourceConfig(
            "file",
            str(tmp_path / "missing.avi"),
        )
    )

    with pytest.raises(FileNotFoundError):
        pipeline.run()
