from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.camera_management import registry
from app.main import app


API_KEY = "test-only-secret-key-37450"
URL = "/api/v1/live/start"


@pytest.fixture
def isolated_camera(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYGUARD_CAMERA_REGISTRY",
        str(tmp_path / "camera_profiles.json"),
    )
    registry.cache_clear()

    try:
        registry().register(
            camera_id="CAM-FUSION-001",
            source_type="usb",
            stream={"width_px": 640, "height_px": 480},
        )
        yield
    finally:
        registry.cache_clear()


@pytest.mark.parametrize(
    "fusion_enabled",
    [False, True],
)
def test_live_fusion_mode_wiring(
    isolated_camera,
    fusion_enabled,
):
    with (
        patch(
            "app.api.live_camera.MODEL_PATH"
        ) as model_path,
        patch(
            "app.api.live_camera.YOLODetectionEngine"
        ) as yolo_class,
        patch(
            "app.api.live_camera.MotionDetectionEngine"
        ) as motion_class,
        patch(
            "app.api.live_camera.AirborneFusionDetectionEngine"
        ) as fusion_class,
        patch(
            "app.api.live_camera.AdvancedTrackingAdapter"
        ) as tracker_class,
        patch(
            "app.api.live_camera.manager.start"
        ) as manager_start,
    ):
        model_path.is_file.return_value = True

        with TestClient(
            app,
            headers={"X-API-Key": API_KEY},
        ) as client:
            response = client.post(
                URL,
                json={
                    "camera_id": "CAM-FUSION-001",
                    "source_type": "usb",
                    "source_location": 0,
                    "fusion_enabled": fusion_enabled,
                },
            )

        assert response.status_code == 200, response.text
        yolo_class.assert_called_once()
        manager_start.assert_called_once()

        args, kwargs = manager_start.call_args

        assert len(args) == 2
        assert kwargs["camera_id"] == "CAM-FUSION-001"

        if fusion_enabled:
            motion_class.assert_called_once()
            fusion_class.assert_called_once_with(
                specialized_detector=yolo_class.return_value,
                candidate_detector=motion_class.return_value,
            )
            tracker_class.assert_called_once()

            assert args[1] is fusion_class.return_value
            assert kwargs["tracker"] is tracker_class.return_value
        else:
            motion_class.assert_not_called()
            fusion_class.assert_not_called()
            tracker_class.assert_not_called()

            assert args[1] is yolo_class.return_value
            assert "tracker" not in kwargs


def test_default_mode_does_not_activate_fusion(isolated_camera):
    with (
        patch(
            "app.api.live_camera.MODEL_PATH"
        ) as model_path,
        patch(
            "app.api.live_camera.YOLODetectionEngine"
        ) as yolo_class,
        patch(
            "app.api.live_camera.MotionDetectionEngine"
        ) as motion_class,
        patch(
            "app.api.live_camera.AirborneFusionDetectionEngine"
        ) as fusion_class,
        patch(
            "app.api.live_camera.manager.start"
        ) as manager_start,
    ):
        model_path.is_file.return_value = True

        with TestClient(
            app,
            headers={"X-API-Key": API_KEY},
        ) as client:
            response = client.post(
                URL,
                json={
                    "camera_id": "CAM-FUSION-001",
                    "source_type": "usb",
                    "source_location": 0,
                },
            )

        assert response.status_code == 200, response.text
        manager_start.assert_called_once()
        yolo_class.assert_called_once()
        motion_class.assert_not_called()
        fusion_class.assert_not_called()


def test_invalid_camera_rejected_before_fusion_creation(
    isolated_camera,
):
    with (
        patch(
            "app.api.live_camera.YOLODetectionEngine"
        ) as yolo_class,
        patch(
            "app.api.live_camera.MotionDetectionEngine"
        ) as motion_class,
        patch(
            "app.api.live_camera.AirborneFusionDetectionEngine"
        ) as fusion_class,
        patch(
            "app.api.live_camera.manager.start"
        ) as manager_start,
    ):
        with TestClient(
            app,
            headers={"X-API-Key": API_KEY},
        ) as client:
            response = client.post(
                URL,
                json={
                    "camera_id": "NONEXISTENT",
                    "source_type": "usb",
                    "fusion_enabled": True,
                },
            )

        assert response.status_code == 404
        yolo_class.assert_not_called()
        motion_class.assert_not_called()
        fusion_class.assert_not_called()
        manager_start.assert_not_called()


def test_fusion_start_requires_api_key(isolated_camera):
    with (
        patch(
            "app.api.live_camera.YOLODetectionEngine"
        ) as yolo_class,
        patch(
            "app.api.live_camera.manager.start"
        ) as manager_start,
    ):
        with TestClient(app) as client:
            response = client.post(
                URL,
                json={
                    "camera_id": "CAM-FUSION-001",
                    "source_type": "usb",
                    "fusion_enabled": True,
                },
            )

        assert response.status_code in (401, 403)
        yolo_class.assert_not_called()
        manager_start.assert_not_called()
