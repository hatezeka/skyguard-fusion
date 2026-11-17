import cv2
import numpy as np
import pytest

from app.detection.pipeline import DetectionPipeline
from app.video.source import VideoSourceConfig


def make_moving_video(path, frames=10):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (160, 120),
    )

    assert writer.isOpened()

    try:
        for index in range(frames):
            image = np.zeros(
                (120, 160, 3),
                dtype=np.uint8,
            )

            x = 10 + index * 8
            image[40:60, x:x + 20] = 255

            writer.write(image)
    finally:
        writer.release()


def test_detection_pipeline_processes_video(tmp_path):
    path = tmp_path / "moving.avi"
    make_moving_video(path)

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", str(path))
    )

    result = pipeline.run()

    assert result.frames_processed == 10
    assert result.total_detections > 0
    assert result.frames_with_detections > 0


def test_detection_pipeline_callback(tmp_path):
    path = tmp_path / "moving.avi"
    make_moving_video(path)

    collected = []

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", str(path))
    )

    pipeline.run(on_detection=collected.append)

    assert len(collected) == 10
    assert collected[0].frame_index == 0
    assert collected[-1].frame_index == 9


def test_detection_pipeline_frame_limit(tmp_path):
    path = tmp_path / "moving.avi"
    make_moving_video(path)

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", str(path))
    )

    result = pipeline.run(max_frames=4)

    assert result.frames_processed == 4


def test_detection_pipeline_reset_between_runs(tmp_path):
    path = tmp_path / "moving.avi"
    make_moving_video(path)

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", str(path))
    )

    first = pipeline.run()
    second = pipeline.run()

    assert first.frames_processed == second.frames_processed
    assert first.total_detections == second.total_detections


def test_detection_pipeline_missing_file(tmp_path):
    pipeline = DetectionPipeline(
        VideoSourceConfig(
            "file",
            str(tmp_path / "missing.avi"),
        )
    )

    with pytest.raises(FileNotFoundError):
        pipeline.run()


def test_detection_pipeline_invalid_limit(tmp_path):
    path = tmp_path / "moving.avi"
    make_moving_video(path)

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", str(path))
    )

    with pytest.raises(ValueError, match="max_frames"):
        pipeline.run(max_frames=0)
