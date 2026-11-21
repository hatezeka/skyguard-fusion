from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.camera_management import registry
from app.detection.live_session import LiveSessionManager


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYGUARD_CAMERA_REGISTRY",
        str(tmp_path / "camera_profiles.json"),
    )
    monkeypatch.delenv("SKYGUARD_ENABLE_RTSP_LIVE", raising=False)
    registry.cache_clear()

    manager = LiveSessionManager()

    try:
        with (
            patch("app.api.live_camera.manager", manager),
            patch("app.api.live_camera.MODEL_PATH") as model_path,
            patch(
                "app.api.live_camera.YOLODetectionEngine",
                new_callable=MagicMock,
            ) as detector,
            TestClient(app, headers={'X-API-Key': 'test-only-secret-key-37450'}) as client,
        ):
            model_path.is_file.return_value = True
            yield client, detector
    finally:
        manager.stop()
        registry.cache_clear()


def register(client, camera_id, source_type, location):
    return client.post(
        "/api/v1/camera-profiles",
        json={
            "camera_id": camera_id,
            "source_type": source_type,
            "source_location": location,
            "stream": {
                "width_px": 640,
                "height_px": 480,
                "fps": 30,
            },
        },
    )


def test_unknown_camera_rejected_before_detector(environment):
    client, detector = environment

    response = client.post(
        "/api/v1/live/start",
        json={
            "camera_id": "UNKNOWN-CAMERA",
            "source_type": "usb",
            "source_location": 0,
        },
    )

    assert response.status_code == 404
    detector.assert_not_called()


def test_wrong_usb_binding_rejected_before_detector(environment):
    client, detector = environment

    assert register(client, "SEC-USB", "usb", 0).status_code == 201

    response = client.post(
        "/api/v1/live/start",
        json={
            "camera_id": "SEC-USB",
            "source_type": "usb",
            "source_location": 1,
        },
    )

    assert response.status_code == 422
    detector.assert_not_called()


def test_wrong_source_type_rejected_before_detector(environment):
    client, detector = environment

    assert register(client, "SEC-TYPE", "usb", 0).status_code == 201

    response = client.post(
        "/api/v1/live/start",
        json={
            "camera_id": "SEC-TYPE",
            "source_type": "rtsp",
            "source_location": "rtsp://camera.local/live",
        },
    )

    assert response.status_code == 422
    detector.assert_not_called()


def test_rtsp_credentials_rejected_without_leak(environment):
    client, detector = environment

    response = register(
        client,
        "SEC-RTSP",
        "rtsp",
        "rtsp://admin:secret@camera.local/live",
    )

    assert response.status_code == 422
    assert "secret" not in response.text
    detector.assert_not_called()


def test_rtsp_disabled_blocks_start(environment):
    client, detector = environment

    assert register(
        client,
        "SEC-RTSP-DISABLED",
        "rtsp",
        "rtsp://camera.local/live",
    ).status_code == 201

    response = client.post(
        "/api/v1/live/start",
        json={
            "camera_id": "SEC-RTSP-DISABLED",
            "source_type": "rtsp",
            "source_location": "rtsp://camera.local/live",
        },
    )

    assert response.status_code == 403
    assert "RTSP live input is disabled" in response.text
