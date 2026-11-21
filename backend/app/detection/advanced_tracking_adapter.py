from dataclasses import dataclass

from app.detection.engine import Detection
from app.detection.motion_aware_v2 import MotionAwareTrackerV2


@dataclass(frozen=True)
class AdvancedTrackObservation:
    track_id: int
    detection: Detection
    frame_index: int
    state: str = "confirmed"


@dataclass(frozen=True)
class AmbiguousTrackObservation:
    detection: Detection
    frame_index: int
    candidate_track_ids: tuple[int, ...]
    state: str = "ambiguous"


@dataclass(frozen=True)
class AdvancedTrackingResult:
    confirmed: list[AdvancedTrackObservation]
    ambiguous: list[AmbiguousTrackObservation]


class AdvancedTrackingAdapter:
    def __init__(self, tracker=None):
        self.tracker = (
            tracker
            if tracker is not None
            else MotionAwareTrackerV2()
        )

    def reset(self):
        self.tracker.reset()

    def update(self, frame_index, detections):
        result = self.tracker.update(
            frame_index=frame_index,
            detections=detections,
        )

        confirmed = [
            AdvancedTrackObservation(
                track_id=observation.track_id,
                detection=observation.detection,
                frame_index=frame_index,
            )
            for observation in result.confirmed
        ]

        ambiguous = [
            AmbiguousTrackObservation(
                detection=observation.detection,
                frame_index=frame_index,
                candidate_track_ids=(
                    observation.candidate_track_ids
                ),
            )
            for observation in result.ambiguous
        ]

        return AdvancedTrackingResult(
            confirmed=confirmed,
            ambiguous=ambiguous,
        )
