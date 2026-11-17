from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib

import cv2
import numpy as np

from app.detection.pipeline import DetectionPipeline
from app.detection.yolo import YOLODetectionEngine
from app.video.source import VideoSourceConfig


def create_video(path: Path, frame_count: int = 5):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 240),
    )

    if not writer.isOpened():
        raise RuntimeError("Cannot create test video")

    try:
        for index in range(frame_count):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)

            cv2.rectangle(
                frame,
                (30 + index * 12, 80),
                (80 + index * 12, 130),
                (255, 255, 255),
                -1,
            )

            writer.write(frame)
    finally:
        writer.release()


def main():
    model_path = Path("models/yolo11n.pt")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

    print("Model:", model_path)
    print("SHA256:", digest)

    detector = YOLODetectionEngine(
        model_path=str(model_path),
        confidence_threshold=0.25,
    )

    names = detector.model.names
    classes = set(names.values()) if isinstance(names, dict) else set(names)

    print("Supports bird:", "bird" in classes)
    print("Supports drone:", "drone" in classes)

    frames = []

    with TemporaryDirectory() as directory:
        video_path = Path(directory) / "real_inference.avi"

        create_video(video_path)

        pipeline = DetectionPipeline(
            VideoSourceConfig("file", str(video_path)),
            profile="standard",
            detector=detector,
        )

        result = pipeline.run(
            on_detection=frames.append,
            max_frames=5,
        )

    print("\n=== REAL YOLO VIDEO PIPELINE ===")
    print("Frames processed:", result.frames_processed)
    print("Total detections:", result.total_detections)
    print("Frames with detections:", result.frames_with_detections)
    print("Processing FPS:", round(result.processing_fps, 2))

    for frame in frames:
        print(
            f"Frame {frame.frame_index}: "
            f"{len(frame.detections)} detections"
        )

        for detection in frame.detections:
            print(
                "  Class:", detection.object_class,
                "Confidence:", round(detection.confidence, 4),
                "Box:", detection.bbox,
            )

    assert result.frames_processed == 5
    assert len(frames) == 5
    assert [frame.frame_index for frame in frames] == list(range(5))
    assert result.processing_fps > 0

    print("\nREAL VIDEO INFERENCE: PASSED")


if __name__ == "__main__":
    main()
