import time
from unittest.mock import patch, MagicMock

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


def wait_for_stopped(client):
    deadline = time.monotonic() + 3

    while time.monotonic() < deadline:
        response = client.get("/api/v1/live/status")
        assert response.status_code == 200

        if response.json()["state"] == "stopped":
            return response.json()

        time.sleep(0.01)

    raise AssertionError("Live session did not stop")


def test_register_start_status_stop_http(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYGUARD_CAMERA_REGISTRY",
        str(tmp_path / "camera_profiles.json"),
    )
    registry.cache_clear()

    manager = LiveSessionManager()

    try:
        with (
            patch("app.api.live_camera.manager", manager),
            patch("app.api.live_camera.MODEL_PATH") as model_path,
            patch(
                "app.api.live_camera.YOLODetectionEngine",
                new_callable=MagicMock,
            ) as detector_class,
            patch(
                "app.detection.live_session.UniversalVideoSource",
                EmptyVideoSource,
            ),
        ):
            model_path.is_file.return_value = True

            with TestClient(app, headers={'X-API-Key': 'test-only-secret-key-37450'}) as client:
                registration = client.post(
                    "/api/v1/camera-profiles",
                    json={
                        "camera_id": "CAM-E2E-001",
                        "source_type": "usb",
                        "source_location": 0,
                        "stream": {
                            "width_px": 640,
                            "height_px": 480,
                            "fps": 30,
                        },
                    },
                )

                assert registration.status_code == 201
                assert registration.json()["source_location"] == 0

                wrong_source = client.post(
                    "/api/v1/live/start",
                    json={
                        "camera_id": "CAM-E2E-001",
                        "source_type": "usb",
                        "source_location": 1,
                    },
                )

                assert wrong_source.status_code == 422
                detector_class.assert_not_called()

                started = client.post(
                    "/api/v1/live/start",
                    json={
                        "camera_id": "CAM-E2E-001",
                        "source_type": "usb",
                        "source_location": 0,
                    },
                )

                assert started.status_code == 200
                assert started.json()["camera_id"] == "CAM-E2E-001"

                status = client.get("/api/v1/live/status")
                assert status.status_code == 200
                assert status.json()["camera_id"] == "CAM-E2E-001"

                stopped = client.post("/api/v1/live/stop")
                assert stopped.status_code == 200
                assert stopped.json()["camera_id"] == "CAM-E2E-001"

                final_status = wait_for_stopped(client)
                assert final_status["camera_id"] == "CAM-E2E-001"

    finally:
        manager.stop()
        registry.cache_clear()
