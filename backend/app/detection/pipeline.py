from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from app.detection.engine import Detection, DetectionEngine, DetectionValidator
from app.detection.motion import MotionDetectionEngine
from app.video.pipeline import (
    ProcessedFrame,
    VideoProcessingPipeline,
)
from app.video.source import VideoSourceConfig


@dataclass(frozen=True)
class DetectionFrame:
    frame_index: int
    detections: list[Detection]
    width: int
    height: int


@dataclass(frozen=True)
class DetectionPipelineResult:
    frames_processed: int
    total_detections: int
    frames_with_detections: int
    elapsed_seconds: float
    processing_fps: float


class DetectionPipeline:
    def __init__(
        self,
        config: VideoSourceConfig,
        profile: str = "standard",
        detector: DetectionEngine | None = None,
    ):
        self.config = config
        self.profile = profile
        self.detector = detector or MotionDetectionEngine()

    def run(
        self,
        on_detection: Callable[[DetectionFrame], None] | None = None,
        max_frames: int | None = None,
    ) -> DetectionPipelineResult:
        reset = getattr(self.detector, "reset", None)
        if callable(reset):
            reset()

        total_detections = 0
        frames_with_detections = 0

        started = perf_counter()

        def process_frame(frame: ProcessedFrame):
            nonlocal total_detections
            nonlocal frames_with_detections

            detections = self.detector.detect(frame.image)

            DetectionValidator.validate(
                frame.image,
                detections,
            )

            total_detections += len(detections)

            if detections:
                frames_with_detections += 1

            if on_detection is not None:
                on_detection(
                    DetectionFrame(
                        frame_index=frame.index,
                        detections=detections,
                        width=frame.width,
                        height=frame.height,
                    )
                )

        pipeline = VideoProcessingPipeline(
            self.config,
            profile=self.profile,
        )

        result = pipeline.run(
            on_frame=process_frame,
            max_frames=max_frames,
        )

        elapsed = perf_counter() - started

        return DetectionPipelineResult(
            frames_processed=result.frames_processed,
            total_detections=total_detections,
            frames_with_detections=frames_with_detections,
            elapsed_seconds=elapsed,
            processing_fps=(
                result.frames_processed / elapsed
                if elapsed > 0
                else 0.0
            ),
        )
