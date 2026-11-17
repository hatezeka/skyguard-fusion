import numpy as np
import pytest

from app.detection.engine import (
    BoundingBox,
    Detection,
    DetectionEngine,
    DetectionValidator,
)


@pytest.fixture
def frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_valid_detection(frame):
    detection = Detection(
        "drone",
        0.95,
        BoundingBox(10, 20, 30, 40),
    )

    result = DetectionValidator.validate(
        frame, [detection]
    )

    assert result == [detection]


def test_multiple_detections(frame):
    detections = [
        Detection("drone", 0.9, BoundingBox(1, 2, 20, 20)),
        Detection("bird", 0.8, BoundingBox(50, 50, 30, 30)),
    ]

    assert len(
        DetectionValidator.validate(frame, detections)
    ) == 2


def test_invalid_confidence(frame):
    detection = Detection(
        "drone", 1.5, BoundingBox(1, 1, 10, 10)
    )

    with pytest.raises(ValueError, match="Confidence"):
        DetectionValidator.validate(frame, [detection])


def test_invalid_bbox(frame):
    detection = Detection(
        "drone", 0.8, BoundingBox(150, 100, 30, 30)
    )

    with pytest.raises(ValueError, match="exceeds"):
        DetectionValidator.validate(frame, [detection])


def test_invalid_class(frame):
    detection = Detection(
        "airplane", 0.8, BoundingBox(1, 1, 10, 10)
    )

    with pytest.raises(ValueError, match="class"):
        DetectionValidator.validate(frame, [detection])


def test_nonfinite_confidence(frame):
    detection = Detection(
        "bird", float("nan"), BoundingBox(1, 1, 10, 10)
    )

    with pytest.raises(ValueError, match="Non-finite"):
        DetectionValidator.validate(frame, [detection])


def test_invalid_image():
    image = np.zeros((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid image"):
        DetectionValidator.validate(image, [])


def test_engine_requires_implementation(frame):
    engine = DetectionEngine()

    with pytest.raises(NotImplementedError):
        engine.detect(frame)
