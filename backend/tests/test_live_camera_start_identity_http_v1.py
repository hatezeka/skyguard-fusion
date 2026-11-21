from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.camera_management import registry


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYGUARD_CAMERA_REGISTRY",
        str(tmp_path / "camera_profiles.json"),
    )
    registry.cache_clear()

    try:
        registry().register(
            camera_id="CAM-001",
            source_type="usb",
            stream={"width_px": 640, "height_px": 480},
        )
        yield
    finally:
        registry.cache_clear()


@pytest.mark.parametrize(
    "payload,expected_status",
    [
        ({"camera_id": "UNKNOWN", "source_type": "usb"}, 404),
        ({"camera_id": "CAM-001", "source_type": "rtsp"}, 422),
        ({"camera_id": "../CAM", "source_type": "usb"}, 422),
        ({"camera_id": "", "source_type": "usb"}, 422),
    ],
)
def test_start_rejects_invalid_identity_before_model(
    isolated_registry,
    payload,
    expected_status,
):
    with (
        patch(
            "app.api.live_camera.MODEL_PATH"
        ) as model_path,
        patch(
            "app.api.live_camera.YOLODetectionEngine"
        ) as detector,
        patch(
            "app.api.live_camera.manager.start"
        ) as manager_start,
    ):
        with TestClient(app, headers={'X-API-Key': 'test-only-secret-key-37450'}) as client:
            response = client.post(
                "/api/v1/live/start",
                json=payload,
            )

        assert response.status_code == expected_status
        model_path.is_file.assert_not_called()
        detector.assert_not_called()
        manager_start.assert_not_called()


def test_valid_identity_reaches_model_check(isolated_registry):
    with (
        patch(
            "app.api.live_camera.MODEL_PATH"
        ) as model_path,
        patch(
            "app.api.live_camera.YOLODetectionEngine"
        ) as detector,
        patch(
            "app.api.live_camera.manager.start"
        ) as manager_start,
    ):
        model_path.is_file.return_value = False

        with TestClient(app, headers={'X-API-Key': 'test-only-secret-key-37450'}) as client:
            response = client.post(
                "/api/v1/live/start",
                json={
                    "camera_id": "CAM-001",
                    "source_type": "usb",
                    "source_location": 0,
                },
            )

        assert response.status_code == 503
        model_path.is_file.assert_called_once()
        detector.assert_not_called()
        manager_start.assert_not_called()
