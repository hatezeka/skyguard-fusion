from app.detection.live_session import LiveSessionManager, SessionSnapshot
from app.api import live_camera


def make_snapshot(state, index):
    return SessionSnapshot(state, index, [], [], None, None)


def test_atomic_frame_pair():
    manager = LiveSessionManager()
    expected = make_snapshot("running", 42)

    with manager._lock:
        manager._snapshot = expected
        manager._jpeg = b"jpeg-42"

    snapshot, jpeg = manager.latest_frame()

    assert snapshot is expected
    assert snapshot.frame_index == 42
    assert jpeg == b"jpeg-42"


def test_mjpeg_uses_atomic_read(monkeypatch):
    class FakeManager:
        calls = 0

        def latest_frame(self):
            self.calls += 1
            if self.calls == 1:
                return make_snapshot("running", 7), b"jpeg-7"
            return make_snapshot("stopped", 7), b"jpeg-7"

        def status(self):
            raise AssertionError("Separate status read")

        def latest_jpeg(self):
            raise AssertionError("Separate JPEG read")

    fake = FakeManager()

    monkeypatch.setattr(live_camera, "manager", fake)
    monkeypatch.setattr(live_camera.time, "sleep", lambda _: None)

    parts = list(live_camera.mjpeg_frames())

    assert len(parts) == 1
    assert b"jpeg-7" in parts[0]
    assert fake.calls == 2
