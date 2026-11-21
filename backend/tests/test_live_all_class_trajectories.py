import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.detection.advanced_tracking_adapter import AdvancedTrackingAdapter
from app.detection.engine import BoundingBox, Detection
from app.detection.live_session import LiveSessionManager


def make_detection(object_class, x):
    return Detection(
        object_class=object_class,
        confidence=0.9,
        bbox=BoundingBox(
            x=float(x),
            y=40.0,
            width=20.0,
            height=20.0,
        ),
    )


class FakeDetector:
    def detect(self, image):
        return [
            make_detection("drone", 20),
            make_detection("bird", 100),
            make_detection("unknown", 200),
        ]


class FakeSource:
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
            image=np.zeros((240, 320, 3), dtype=np.uint8),
        )

        self.index += 1
        return frame


def wait_for_stopped(manager):
    deadline = time.monotonic() + 3.0

    while time.monotonic() < deadline:
        snapshot = manager.status()

        if snapshot.state in ("stopped", "error"):
            return snapshot

        time.sleep(0.01)

    raise AssertionError("Session timeout")


def test_trajectories_include_drone_bird_and_unknown():
    manager = LiveSessionManager()

    with patch(
        "app.detection.live_session.UniversalVideoSource",
        FakeSource,
    ):
        manager.start(
            config=object(),
            detector=FakeDetector(),
            tracker=AdvancedTrackingAdapter(),
        )

        snapshot = wait_for_stopped(manager)

    assert snapshot.state == "stopped", snapshot.error

    classes = {
        trajectory["object_class"]
        for trajectory in snapshot.trajectories
    }

    assert classes == {"drone", "bird", "unknown"}

    assert len(snapshot.trajectories) == 3

    assert all(
        len(trajectory["points"]) == 1
        for trajectory in snapshot.trajectories
    )
