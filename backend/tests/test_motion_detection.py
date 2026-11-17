import numpy as np
import pytest

from app.detection.motion import MotionDetectionEngine


def make_frame(x=None):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    if x is not None:
        frame[40:60, x:x + 20] = 255

    return frame


def test_first_frame_has_no_detection():
    engine = MotionDetectionEngine()

    assert engine.detect(make_frame(10)) == []


def test_moving_object_detected():
    engine = MotionDetectionEngine()

    engine.detect(make_frame(10))
    detections = engine.detect(make_frame(50))

    assert len(detections) >= 1
    assert all(
        detection.object_class == "unknown"
        for detection in detections
    )


def test_static_scene_no_detection():
    engine = MotionDetectionEngine()

    frame = make_frame(20)

    engine.detect(frame)
    detections = engine.detect(frame.copy())

    assert detections == []


def test_reset_clears_previous_frame():
    engine = MotionDetectionEngine()

    engine.detect(make_frame(10))
    engine.reset()

    assert engine.detect(make_frame(50)) == []


def test_invalid_min_area():
    with pytest.raises(ValueError, match="min_area"):
        MotionDetectionEngine(min_area=0)


def test_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        MotionDetectionEngine(threshold=256)


def test_invalid_frame():
    engine = MotionDetectionEngine()

    with pytest.raises(ValueError, match="Invalid image"):
        engine.detect(
            np.zeros((0, 0, 3), dtype=np.uint8)
        )


def test_multiple_moving_objects():
    engine = MotionDetectionEngine()

    first = np.zeros((120, 160, 3), dtype=np.uint8)
    second = first.copy()

    second[20:40, 20:40] = 255
    second[70:90, 100:120] = 255

    engine.detect(first)
    detections = engine.detect(second)

    assert len(detections) == 2
