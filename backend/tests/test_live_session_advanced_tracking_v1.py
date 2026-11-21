import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.detection.advanced_tracking_adapter import (
    AdvancedTrackingAdapter,
)
from app.detection.engine import BoundingBox, Detection
from app.detection.live_session import LiveSessionManager


def make_detection(x, object_class="drone"):
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
            make_detection(30.0),
            make_detection(120.0, "unknown"),
        ]


class FakeVideoSource:
    def __init__(self, config):
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if self.index >= 1:
            return None

        frame = SimpleNamespace(
            index=self.index,
            image=np.zeros(
                (240, 320, 3),
                dtype=np.uint8,
            ),
        )

        self.index += 1
        return frame


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
                ),
            ],
            ambiguous=[
                SimpleNamespace(
                    detection_index=1,
                    detection=detections[1],
                    candidate_track_ids=(8, 9),
                ),
            ],
        )


def wait_for_stopped(manager, timeout=3):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        snapshot = manager.status()

        if snapshot.state in ("stopped", "error"):
            return snapshot

        time.sleep(0.01)

    raise AssertionError("Session did not finish")


def run_session(tracker=None):
    manager = LiveSessionManager()

    with patch(
        "app.detection.live_session.UniversalVideoSource",
        FakeVideoSource,
    ):
        manager.start(
            config=object(),
            detector=FakeDetector(),
            camera_id="CAM-ADV-001",
            tracker=tracker,
        )

        snapshot = wait_for_stopped(manager)

    assert snapshot.state == "stopped", snapshot.error

    return manager, snapshot


def test_advanced_tracking_preserves_confirmed_identity():
    _, snapshot = run_session(
        tracker=AdvancedTrackingAdapter(
            tracker=ControlledTracker(),
        ),
    )

    assert snapshot.camera_id == "CAM-ADV-001"
    assert len(snapshot.tracks) == 1
    assert snapshot.tracks[0]["track_id"] == 7
    assert snapshot.tracks[0]["state"] == "confirmed"


def test_ambiguous_identity_is_reported_separately():
    _, snapshot = run_session(
        tracker=AdvancedTrackingAdapter(
            tracker=ControlledTracker(),
        ),
    )

    assert len(snapshot.ambiguous_tracks) == 1

    ambiguous = snapshot.ambiguous_tracks[0]

    assert ambiguous["state"] == "ambiguous"
    assert ambiguous["candidate_track_ids"] == [8, 9]
    assert "track_id" not in ambiguous


def test_unknown_class_remains_unknown():
    _, snapshot = run_session(
        tracker=AdvancedTrackingAdapter(
            tracker=ControlledTracker(),
        ),
    )

    assert snapshot.ambiguous_tracks[0]["class"] == "unknown"


def test_advanced_tracking_produces_jpeg():
    manager, snapshot = run_session(
        tracker=AdvancedTrackingAdapter(
            tracker=ControlledTracker(),
        ),
    )

    jpeg = manager.latest_jpeg()

    assert isinstance(jpeg, bytes)
    assert jpeg.startswith(bytes([0xFF, 0xD8]))
    assert snapshot.frame_index == 0


def test_legacy_session_still_works():
    _, snapshot = run_session()

    assert snapshot.camera_id == "CAM-ADV-001"
    assert len(snapshot.tracks) == 2
    assert snapshot.ambiguous_tracks == []
