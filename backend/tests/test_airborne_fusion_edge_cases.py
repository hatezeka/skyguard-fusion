import numpy as np
import pytest

from app.detection.airborne_fusion import AirborneFusionDetectionEngine
from app.detection.engine import BoundingBox, Detection


class FakeDetector:
    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, image):
        return list(self.detections)


def detection(object_class, x, y, w=20, h=20, confidence=0.9):
    return Detection(
        object_class=object_class,
        confidence=confidence,
        bbox=BoundingBox(x, y, w, h),
    )


@pytest.fixture
def frame():
    return np.zeros((200, 300, 3), dtype=np.uint8)


def fuse(frame, specialized, candidates):
    engine = AirborneFusionDetectionEngine(
        FakeDetector(specialized),
        FakeDetector(candidates),
    )
    return engine.detect(frame)


def test_multiple_separate_unknown_objects(frame):
    result = fuse(
        frame,
        [],
        [
            detection("unknown", 10, 10),
            detection("unknown", 100, 20),
            detection("unknown", 200, 100),
        ],
    )

    assert len(result) == 3
    assert all(d.object_class == "unknown" for d in result)


def test_drone_bird_and_unknown_together(frame):
    result = fuse(
        frame,
        [
            detection("drone", 10, 10),
            detection("bird", 100, 20),
        ],
        [detection("unknown", 200, 100)],
    )

    assert len(result) == 3
    assert {d.object_class for d in result} == {
        "drone", "bird", "unknown"
    }


def test_unknown_overlapping_drone_not_duplicated(frame):
    result = fuse(
        frame,
        [detection("drone", 40, 40, 40, 40)],
        [detection("unknown", 45, 45, 40, 40)],
    )

    assert len(result) == 1
    assert result[0].object_class == "drone"


def test_duplicate_unknown_candidates_are_suppressed(frame):
    result = fuse(
        frame,
        [],
        [
            detection("unknown", 40, 40, 30, 30),
            detection("unknown", 42, 42, 30, 30),
        ],
    )

    assert len(result) == 1


def test_unknown_far_from_drone_is_preserved(frame):
    result = fuse(
        frame,
        [detection("drone", 20, 20)],
        [detection("unknown", 200, 150)],
    )

    assert len(result) == 2


def test_non_unknown_candidate_does_not_override_yolo(frame):
    result = fuse(
        frame,
        [detection("bird", 20, 20)],
        [detection("drone", 20, 20)],
    )

    assert len(result) == 1
    assert result[0].object_class == "bird"


def test_invalid_frame_rejected():
    engine = AirborneFusionDetectionEngine(
        FakeDetector([]),
        FakeDetector([]),
    )

    with pytest.raises(ValueError, match="Invalid image"):
        engine.detect(np.zeros((0, 0, 3), dtype=np.uint8))
