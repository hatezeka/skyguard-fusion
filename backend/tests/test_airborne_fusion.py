import numpy as np
import pytest

from app.detection.airborne_fusion import AirborneFusionDetectionEngine
from app.detection.engine import BoundingBox, Detection


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.reset_count = 0

    def detect(self, image):
        return list(self.detections)

    def reset(self):
        self.reset_count += 1


def make_detection(object_class, x, y, confidence=0.9):
    return Detection(
        object_class=object_class,
        confidence=confidence,
        bbox=BoundingBox(x, y, 20, 20),
    )


@pytest.fixture
def image():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_drone_preserved_when_motion_overlaps(image):
    drone = make_detection("drone", 10, 10)
    motion = make_detection("unknown", 10, 10, 0.5)

    engine = AirborneFusionDetectionEngine(
        FakeDetector([drone]),
        FakeDetector([motion]),
    )

    result = engine.detect(image)

    assert len(result) == 1
    assert result[0].object_class == "drone"


def test_bird_preserved_when_motion_overlaps(image):
    bird = make_detection("bird", 30, 30)
    motion = make_detection("unknown", 30, 30, 0.5)

    engine = AirborneFusionDetectionEngine(
        FakeDetector([bird]),
        FakeDetector([motion]),
    )

    result = engine.detect(image)

    assert len(result) == 1
    assert result[0].object_class == "bird"


def test_separate_unknown_object_is_retained(image):
    drone = make_detection("drone", 10, 10)
    unknown = make_detection("unknown", 100, 70, 0.5)

    engine = AirborneFusionDetectionEngine(
        FakeDetector([drone]),
        FakeDetector([unknown]),
    )

    result = engine.detect(image)

    assert len(result) == 2
    assert {d.object_class for d in result} == {"drone", "unknown"}


def test_empty_scene_has_no_detection(image):
    engine = AirborneFusionDetectionEngine(
        FakeDetector([]),
        FakeDetector([]),
    )

    assert engine.detect(image) == []


def test_reset_both_detectors():
    specialized = FakeDetector([])
    candidates = FakeDetector([])

    engine = AirborneFusionDetectionEngine(
        specialized,
        candidates,
    )

    engine.reset()

    assert specialized.reset_count == 1
    assert candidates.reset_count == 1


def test_invalid_overlap_threshold():
    with pytest.raises(ValueError):
        AirborneFusionDetectionEngine(
            FakeDetector([]),
            FakeDetector([]),
            overlap_threshold=0,
        )
