from dataclasses import dataclass

from app.detection.engine import BoundingBox, Detection
from app.detection.tracker import ObjectTracker
from app.tracking_validation.evaluator import (
    IdentityMetrics,
    evaluate_identity_sequence,
)
from app.tracking_validation.scenarios import SyntheticFrame


@dataclass(frozen=True)
class TrackingRun:
    scenario: str
    frames: int
    objects: int
    metrics: IdentityMetrics
    unique_track_ids: int


def run_scenario(
    scenario_name: str,
    frames: list[SyntheticFrame],
    tracker: ObjectTracker | None = None,
    box_size: float = 20.0,
) -> TrackingRun:
    if not frames:
        raise ValueError("Scenario cannot be empty")

    if box_size <= 0:
        raise ValueError("box_size must be positive")

    tracker = tracker if tracker is not None else ObjectTracker()
    tracker.reset()

    identity_frames = []
    unique_ids = set()
    object_identities = set()

    for frame in frames:
        detections = []
        detection_identity = {}

        for obj in frame.objects:
            detection = Detection(
                object_class=obj.object_class,
                confidence=0.95,
                bbox=BoundingBox(
                    x=obj.x,
                    y=obj.y,
                    width=box_size,
                    height=box_size,
                ),
            )

            detections.append(detection)
            detection_identity[id(detection)] = obj.identity
            object_identities.add(obj.identity)

        observations = tracker.update(
            frame_index=frame.frame_index,
            detections=detections,
        )

        assignments = {
            identity: None
            for identity in detection_identity.values()
        }

        for observation in observations:
            identity = detection_identity.get(
                id(observation.detection)
            )

            if identity is None:
                raise ValueError(
                    "Tracker returned an unknown detection"
                )

            if assignments[identity] is not None:
                raise ValueError(
                    "Duplicate assignment for one identity"
                )

            assignments[identity] = observation.track_id
            unique_ids.add(observation.track_id)

        identity_frames.append(assignments)

    return TrackingRun(
        scenario=scenario_name,
        frames=len(frames),
        objects=len(object_identities),
        metrics=evaluate_identity_sequence(identity_frames),
        unique_track_ids=len(unique_ids),
    )
