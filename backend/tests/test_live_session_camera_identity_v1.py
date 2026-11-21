import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.detection.live_session import LiveSessionManager


class FakeDetector:
    def detect(self, image):
        return []


class ControlledVideoSource:
    entered = threading.Event()
    release = threading.Event()

    def __init__(self, config):
        self.config = config
        self.index = 0

    def __enter__(self):
        self.entered.set()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if self.index == 0:
            self.index += 1
            return SimpleNamespace(
                index=0,
                image=np.zeros((64, 64, 3), dtype=np.uint8),
            )

        self.release.wait(timeout=3)
        return None


def wait_for_state(manager, expected, timeout=3):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        snapshot = manager.status()

        if snapshot.state == expected:
            return snapshot

        time.sleep(0.01)

    raise AssertionError(
        f"Expected {expected}, got {manager.status().state}"
    )


def test_identity_through_running_stopping_and_stopped():
    manager = LiveSessionManager()

    ControlledVideoSource.entered.clear()
    ControlledVideoSource.release.clear()

    with patch(
        "app.detection.live_session.UniversalVideoSource",
        ControlledVideoSource,
    ):
        try:
            manager.start(
                config=object(),
                detector=FakeDetector(),
                camera_id="CAM-001",
            )

            assert manager.status().camera_id == "CAM-001"

            running = wait_for_state(manager, "running")
            assert running.camera_id == "CAM-001"

            started = time.monotonic()
            stopping = manager.stop()
            elapsed = time.monotonic() - started

            assert elapsed < 0.5
            assert stopping.state == "stopping"
            assert stopping.camera_id == "CAM-001"

        finally:
            ControlledVideoSource.release.set()

        stopped = wait_for_state(manager, "stopped")
        assert stopped.camera_id == "CAM-001"


def test_new_session_does_not_inherit_previous_identity():
    manager = LiveSessionManager()

    class EmptyVideoSource:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return None

    with patch(
        "app.detection.live_session.UniversalVideoSource",
        EmptyVideoSource,
    ):
        manager.start(object(), FakeDetector(), camera_id="CAM-001")
        first = wait_for_state(manager, "stopped")
        assert first.camera_id == "CAM-001"

        manager.start(object(), FakeDetector(), camera_id="CAM-002")
        second = wait_for_state(manager, "stopped")
        assert second.camera_id == "CAM-002"

        manager.start(object(), FakeDetector())
        third = wait_for_state(manager, "stopped")
        assert third.camera_id is None


def test_identity_preserved_on_worker_error():
    manager = LiveSessionManager()

    class FailingVideoSource:
        def __init__(self, config):
            pass

        def __enter__(self):
            raise RuntimeError("Simulated camera failure")

        def __exit__(self, exc_type, exc, tb):
            return False

    with patch(
        "app.detection.live_session.UniversalVideoSource",
        FailingVideoSource,
    ):
        manager.start(
            object(),
            FakeDetector(),
            camera_id="CAM-ERROR",
        )

        snapshot = wait_for_state(manager, "error")

        assert snapshot.camera_id == "CAM-ERROR"
        assert "Simulated camera failure" in snapshot.error
