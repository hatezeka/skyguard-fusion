import numpy as np
import pytest

from app.detection.yolo import YOLODetectionEngine


class FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoordinates:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self

    def tolist(self):
        return self.values


class FakeBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = FakeValue(class_id)
        self.conf = FakeValue(confidence)
        self.xyxy = FakeCoordinates(coordinates)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names

    def predict(self, **kwargs):
        return [FakeResult(self.boxes, self.names)]


@pytest.fixture
def frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_drone_classification(frame):
    model = FakeModel(
        [FakeBox(0, 0.9, [10, 20, 40, 60])],
        {0: "drone"},
    )

    detections = YOLODetectionEngine(model=model).detect(frame)

    assert len(detections) == 1
    assert detections[0].object_class == "drone"
    assert detections[0].bbox.width == 30


def test_bird_classification(frame):
    model = FakeModel(
        [FakeBox(1, 0.8, [20, 20, 40, 40])],
        {1: "bird"},
    )

    detections = YOLODetectionEngine(model=model).detect(frame)

    assert detections[0].object_class == "bird"


def test_other_classification(frame):
    model = FakeModel(
        [FakeBox(2, 0.8, [20, 20, 40, 40])],
        {2: "airplane"},
    )

    detections = YOLODetectionEngine(model=model).detect(frame)

    assert detections[0].object_class == "other"


def test_confidence_filter(frame):
    model = FakeModel(
        [FakeBox(0, 0.2, [10, 10, 30, 30])],
        {0: "drone"},
    )

    detector = YOLODetectionEngine(
        model=model,
        confidence_threshold=0.5,
    )

    assert detector.detect(frame) == []


def test_bounding_box_clipping(frame):
    model = FakeModel(
        [FakeBox(0, 0.9, [-10, -10, 200, 150])],
        {0: "drone"},
    )

    detection = YOLODetectionEngine(model=model).detect(frame)[0]

    assert detection.bbox.x == 0
    assert detection.bbox.y == 0
    assert detection.bbox.width == 160
    assert detection.bbox.height == 120


def test_empty_detections(frame):
    model = FakeModel([], {0: "drone"})

    assert YOLODetectionEngine(model=model).detect(frame) == []


def test_missing_model_file():
    with pytest.raises(FileNotFoundError):
        YOLODetectionEngine(
            model_path="nonexistent_model_file_12345.pt"
        )


def test_invalid_threshold():
    with pytest.raises(ValueError, match="confidence"):
        YOLODetectionEngine(
            model=FakeModel([], {}),
            confidence_threshold=1.5,
        )


def test_invalid_image():
    detector = YOLODetectionEngine(
        model=FakeModel([], {})
    )

    with pytest.raises(ValueError, match="Invalid image"):
        detector.detect(
            np.zeros((0, 0, 3), dtype=np.uint8)
        )


def test_unknown_class_id_rejected(frame):
    model = FakeModel(
        [FakeBox(99, 0.9, [10, 10, 30, 30])],
        {0: "drone"},
    )

    with pytest.raises(ValueError, match="class ID"):
        YOLODetectionEngine(model=model).detect(frame)
