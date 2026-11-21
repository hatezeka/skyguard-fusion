from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.detection.advanced_tracking_adapter import (
    AdvancedTrackingAdapter,
    AdvancedTrackingResult,
    AdvancedTrackObservation,
    AmbiguousTrackObservation,
)
from app.detection.engine import BoundingBox, Detection
from app.detection.live_service import LiveDetectionTrackingService


def detection(x, object_class="drone"):
    return Detection(
        object_class=object_class,
        confidence=0.95,
        bbox=BoundingBox(
            x=x,
            y=40.0,
            width=20.0,
            height=20.0,
        ),
    )


class FakeDetector:
    def detect(self, image):
        return [
            detection(30.0),
            detection(120.0, "unknown"),
        ]


class FakeSource:
    def __init__(self, config):
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if self.read_count:
            return None

        self.read_count += 1

        return SimpleNamespace(
            index=0,
            image=np.zeros((240, 320, 3), dtype=np.uint8),
            width=320,
            height=240,
        )


class ControlledTracker:
    def reset(self):
        pass

    def update(self, frame_index, detections):
        return SimpleNamespace(
            confirmed=[
                SimpleNamespace(
                    track_id=7,
                    detection_index=0,
                    detection=detections[0],
                )
            ],
            ambiguous=[
                SimpleNamespace(
                    detection_index=1,
                    detection=detections[1],
                    candidate_track_ids=(8, 9),
                )
            ],
        )


def run_service(tracker):
    service = LiveDetectionTrackingService(
        detector=FakeDetector(),
        tracker=tracker,
    )

    with patch(
        "app.detection.live_service.UniversalVideoSource",
        FakeSource,
    ):
        return service.run(config=object(), max_frames=1)


def test_ambiguous_identity_has_no_assigned_track_id():
    results = run_service(
        AdvancedTrackingAdapter(
            tracker=ControlledTracker(),
        )
    )

    assert len(results) == 1

    result = results[0]

    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 7

    assert len(result.ambiguous_tracks) == 1

    ambiguous = result.ambiguous_tracks[0]

    assert ambiguous.state == "ambiguous"
    assert ambiguous.candidate_track_ids == (8, 9)
    assert not hasattr(ambiguous, "track_id")


def test_unknown_class_is_preserved_when_identity_is_ambiguous():
    results = run_service(
        AdvancedTrackingAdapter(
            tracker=ControlledTracker(),
        )
    )

    ambiguous = results[0].ambiguous_tracks[0]

    assert ambiguous.detection.object_class == "unknown"
    assert ambiguous.state == "ambiguous"

    confirmed = results[0].tracks[0]

    assert confirmed.detection.object_class == "drone"
    assert confirmed.state == "confirmed"
