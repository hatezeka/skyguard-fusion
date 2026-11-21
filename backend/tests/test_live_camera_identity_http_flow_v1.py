import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.camera_management import registry
from app.detection.live_session import LiveSessionManager


class EmptyVideoSource:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return None


class FakeDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        return []


def wait_for_status(client, expected, timeout=3):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        response = client.get("/api/v1/live/status")
        assert response.status_code == 200

        data = response.json()

        if data["state"] == expected:
            return data

        time.sleep(0.01)

    raise AssertionError(
        f"Expected {expected}, got {data['state']}"
    )


@pytest.fixture
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYGUARD_CAMERA_REGISTRY",
        str(tmp_path / "camera_profiles.json"),
    )

    registry.cache_clear()

    registry().register(
        camera_id="CAM-001",
        source_type="usb",
        stream={"width_px": 640, "height_px": 480},
    )

    registry().register(
        camera_id="CAM-002",
        source_type="usb",
        stream={"width_px": 640, "height_px": 480},
    )

    try:
        yield
    finally:
        registry.cache_clear()


def test_http_start_status_camera_identity(isolated_environment):
    manager = LiveSessionManager()

    with (
        patch("app.api.live_camera.manager", manager),
        patch(
            "app.api.live_camera.MODEL_PATH"
        ) as model_path,
        patch(
            "app.api.live_camera.YOLODetectionEngine",
            FakeDetector,
        ),
        patch(
            "app.detection.live_session.UniversalVideoSource",
            EmptyVideoSource,
        ),
    ):
        model_path.is_file.return_value = True

        with TestClient(app, headers={'X-API-Key': 'test-only-secret-key-37450'}) as client:
            for camera_id in ("CAM-001", "CAM-002"):
                response = client.post(
                    "/api/v1/live/start",
                    json={
                        "camera_id": camera_id,
                        "source_type": "usb",
                        "source_location": 0,
                    },
                )

                assert response.status_code == 200
                assert response.json()["camera_id"] == camera_id

                status = wait_for_status(client, "stopped")
                assert status["camera_id"] == camera_id

            legacy = client.post(
                "/api/v1/live/start",
                json={
                    "source_type": "usb",
                    "source_location": 0,
                },
            )

            assert legacy.status_code == 200
            assert legacy.json()["camera_id"] is None

            legacy_status = wait_for_status(client, "stopped")
            assert legacy_status["camera_id"] is None
