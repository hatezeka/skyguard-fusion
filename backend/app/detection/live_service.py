"""SkyGuard live camera detection and tracking."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from app.detection.engine import DetectionValidator
from app.detection.tracker import ObjectTracker, TrackObservation
from app.video.source import UniversalVideoSource, VideoSourceConfig


@dataclass(frozen=True)
class LiveFrameResult:
    frame_index: int
    width: int
    height: int
    detections_count: int
    tracks: list[TrackObservation]
    processing_seconds: float
    ambiguous_tracks: list = field(default_factory=list)


class LiveDetectionTrackingService:
    def __init__(self, detector, tracker=None):
        self.detector = detector
        self.tracker = tracker if tracker is not None else ObjectTracker()

    def run(
        self,
        config: VideoSourceConfig,
        max_frames: int,
        on_result: Callable[[LiveFrameResult], None] | None = None,
    ) -> list[LiveFrameResult]:

        if max_frames <= 0:
            raise ValueError("max_frames must be positive")

        self.tracker.reset()

        reset_detector = getattr(self.detector, "reset", None)
        if callable(reset_detector):
            reset_detector()

        results = []

        with UniversalVideoSource(config) as source:
            for _ in range(max_frames):
                frame = source.read()

                if frame is None:
                    break

                started = perf_counter()

                detections = self.detector.detect(frame.image)
                DetectionValidator.validate(frame.image, detections)

                tracking_result = self.tracker.update(
                    frame_index=frame.index,
                    detections=detections,
                )

                if (
                    hasattr(tracking_result, "confirmed")
                    and hasattr(tracking_result, "ambiguous")
                ):
                    tracks = list(tracking_result.confirmed)
                    ambiguous_tracks = list(tracking_result.ambiguous)
                else:
                    tracks = tracking_result
                    ambiguous_tracks = []

                elapsed = perf_counter() - started

                result = LiveFrameResult(
                    frame_index=frame.index,
                    width=frame.width,
                    height=frame.height,
                    detections_count=len(detections),
                    tracks=tracks,
                    processing_seconds=elapsed,
                    ambiguous_tracks=ambiguous_tracks,
                )

                results.append(result)

                if on_result is not None:
                    on_result(result)

        return results
