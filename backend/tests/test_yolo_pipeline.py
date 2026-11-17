import cv2
import numpy as np
import pytest

from app.detection.pipeline import DetectionPipeline
from app.detection.yolo import YOLODetectionEngine
from app.video.source import VideoSourceConfig


class FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoordinates:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __getitem__(self, index):
        return self

    def tolist(self):
        return self.coordinates


class FakeBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = FakeValue(class_id)
        self.conf = FakeValue(confidence)
        self.xyxy = FakeCoordinates(coordinates)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYOLOModel:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        assert kwargs["source"].dtype == np.uint8
        return [
            FakeResult(
                self.boxes,
                {0: "drone", 1: "bird"},
            )
        ]


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "sample.avi"

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 120),
    )

    if not writer.isOpened():
        pytest.skip("MJPG video writer unavailable")

    try:
        for index in range(6):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(
                frame,
                (10 + index * 5, 30),
                (35 + index * 5, 55),
                (255, 255, 255),
                -1,
            )
            writer.write(frame)
    finally:
        writer.release()

    return str(path)


def test_yolo_pipeline_processes_video(sample_video):
    model = FakeYOLOModel([
        FakeBox(0, 0.90, [10, 20, 40, 60])
    ])

    detector = YOLODetectionEngine(model=model)
    frames = []

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", sample_video),
        detector=detector,
    )

    result = pipeline.run(on_detection=frames.append)

    assert result.frames_processed == 6
    assert result.total_detections == 6
    assert result.frames_with_detections == 6
    assert result.processing_fps > 0
    assert model.calls == 6
    assert [frame.frame_index for frame in frames] == list(range(6))
    assert all(
        frame.detections[0].object_class == "drone"
        for frame in frames
    )


def test_yolo_pipeline_empty_detections(sample_video):
    model = FakeYOLOModel([])
    detector = YOLODetectionEngine(model=model)

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", sample_video),
        detector=detector,
    )

    result = pipeline.run()

    assert result.frames_processed == 6
    assert result.total_detections == 0
    assert result.frames_with_detections == 0
    assert model.calls == 6


def test_yolo_pipeline_respects_max_frames(sample_video):
    model = FakeYOLOModel([
        FakeBox(1, 0.85, [20, 20, 45, 50])
    ])

    detector = YOLODetectionEngine(model=model)

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", sample_video),
        detector=detector,
    )

    result = pipeline.run(max_frames=3)

    assert result.frames_processed == 3
    assert result.total_detections == 3
    assert model.calls == 3


def test_yolo_pipeline_preserves_bounding_boxes(sample_video):
    model = FakeYOLOModel([
        FakeBox(0, 0.95, [-5, 10, 200, 80])
    ])

    detector = YOLODetectionEngine(model=model)
    frames = []

    pipeline = DetectionPipeline(
        VideoSourceConfig("file", sample_video),
        detector=detector,
    )

    pipeline.run(on_detection=frames.append, max_frames=1)

    detection = frames[0].detections[0]

    assert detection.object_class == "drone"
    assert detection.confidence == pytest.approx(0.95)
    assert detection.bbox.x == 0
    assert detection.bbox.y == 10
    assert detection.bbox.width == 160
    assert detection.bbox.height == 70
