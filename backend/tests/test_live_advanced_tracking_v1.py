from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.detection.advanced_tracking_adapter import (
    AdvancedTrackingAdapter,
    AdvancedTrackingResult,
    AmbiguousTrackObservation,
)
from app.detection.engine import BoundingBox, Detection
from app.detection.live_service import LiveDetectionTrackingService


def make_detection(x=30.0):
    return Detection(
        object_class="drone",
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
            make_detection(30.0),
            make_detection(200.0),
        ]


class FakeSource:
    def __init__(self, config):
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if self.index >= 2:
            return None

        frame = SimpleNamespace(
            index=self.index,
            image=np.zeros(
                (240, 320, 3),
                dtype=np.uint8,
            ),
            width=320,
            height=240,
        )

        self.index += 1
        return frame


def test_live_service_supports_advanced_tracking():
    service = LiveDetectionTrackingService(
        detector=FakeDetector(),
        tracker=AdvancedTrackingAdapter(),
    )

    with patch(
        "app.detection.live_service.UniversalVideoSource",
        FakeSource,
    ):
        results = service.run(
            config=object(),
            max_frames=2,
        )

    assert len(results) == 2

    for result in results:
        assert len(result.tracks) == 2
        assert result.ambiguous_tracks == []

        assert all(
            track.state == "confirmed"
            for track in result.tracks
        )


def test_live_service_keeps_legacy_tracker():
    service = LiveDetectionTrackingService(
        detector=FakeDetector(),
    )

    with patch(
        "app.detection.live_service.UniversalVideoSource",
        FakeSource,
    ):
        results = service.run(
            config=object(),
            max_frames=2,
        )

    assert len(results) == 2
    assert results[0].ambiguous_tracks == []
    assert len(results[0].tracks) == 2


def test_advanced_tracking_result_separates_ambiguity():
    observation = AmbiguousTrackObservation(
        detection=make_detection(),
        frame_index=4,
        candidate_track_ids=(1, 2),
    )

    result = AdvancedTrackingResult(
        confirmed=[],
        ambiguous=[observation],
    )

    assert result.confirmed == []
    assert len(result.ambiguous) == 1
    assert result.ambiguous[0].candidate_track_ids == (1, 2)
    assert not hasattr(result.ambiguous[0], "track_id")
