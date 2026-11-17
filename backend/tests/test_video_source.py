import cv2
import numpy as np
import pytest

from app.video.source import (
    UniversalVideoSource,
    VideoSourceConfig,
)


def make_video(path):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (160, 120),
    )

    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")

    try:
        for index in range(5):
            image = np.full(
                (120, 160, 3),
                index * 30,
                dtype=np.uint8,
            )
            writer.write(image)
    finally:
        writer.release()


def test_file_source_reads_frames(tmp_path):
    path = tmp_path / "sample.avi"
    make_video(path)

    config = VideoSourceConfig("file", str(path))

    with UniversalVideoSource(config) as source:
        frames = []

        while True:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)

    assert len(frames) == 5
    assert [frame.index for frame in frames] == list(range(5))
    assert frames[0].width == 160
    assert frames[0].height == 120


def test_missing_file_rejected(tmp_path):
    config = VideoSourceConfig(
        "file", str(tmp_path / "missing.avi")
    )

    with pytest.raises(FileNotFoundError):
        UniversalVideoSource(config).open()


def test_read_before_open_rejected():
    source = UniversalVideoSource(
        VideoSourceConfig("usb", 0)
    )

    with pytest.raises(RuntimeError, match="not open"):
        source.read()


def test_invalid_usb_index():
    source = UniversalVideoSource(
        VideoSourceConfig("usb", -1)
    )

    with pytest.raises(ValueError, match="camera index"):
        source.open()


def test_invalid_rtsp_url():
    source = UniversalVideoSource(
        VideoSourceConfig("rtsp", "http://example.com/video")
    )

    with pytest.raises(ValueError, match="RTSP"):
        source.open()


def test_invalid_source_type():
    source = UniversalVideoSource(
        VideoSourceConfig("unknown", "x")
    )

    with pytest.raises(ValueError, match="Unsupported"):
        source.open()


def test_double_open_rejected(tmp_path):
    path = tmp_path / "sample.avi"
    make_video(path)

    source = UniversalVideoSource(
        VideoSourceConfig("file", str(path))
    )

    try:
        source.open()

        with pytest.raises(RuntimeError, match="already open"):
            source.open()
    finally:
        source.close()


def test_close_is_idempotent():
    source = UniversalVideoSource(
        VideoSourceConfig("usb", 0)
    )

    source.close()
    source.close()

    assert source.capture is None
