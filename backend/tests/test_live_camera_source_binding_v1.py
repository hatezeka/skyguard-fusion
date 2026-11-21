import pytest
from fastapi import HTTPException

from app.api.live_camera import StartRequest, validate_camera_identity
from app.api.camera_management import registry


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYGUARD_CAMERA_REGISTRY",
        str(tmp_path / "cameras.json"),
    )
    registry.cache_clear()

    registry().register(
        camera_id="CAM-USB-001",
        source_type="usb",
        stream={"width_px": 640, "height_px": 480, "fps": 30},
        source_location=0,
    )

    registry().register(
        camera_id="CAM-RTSP-001",
        source_type="rtsp",
        stream={"width_px": 640, "height_px": 480, "fps": 30},
        source_location="rtsp://camera.local/live",
    )

    try:
        yield
    finally:
        registry.cache_clear()


def test_usb_matching_source(isolated_registry):
    request = StartRequest(
        camera_id="CAM-USB-001",
        source_type="usb",
        source_location=0,
    )

    assert validate_camera_identity(request) == "CAM-USB-001"


def test_usb_mismatched_source_rejected(isolated_registry):
    request = StartRequest(
        camera_id="CAM-USB-001",
        source_type="usb",
        source_location=1,
    )

    with pytest.raises(HTTPException) as exc:
        validate_camera_identity(request)

    assert exc.value.status_code == 422


def test_rtsp_matching_source(isolated_registry):
    request = StartRequest(
        camera_id="CAM-RTSP-001",
        source_type="rtsp",
        source_location="rtsp://camera.local/live",
    )

    assert validate_camera_identity(request) == "CAM-RTSP-001"


def test_rtsp_mismatched_source_rejected(isolated_registry):
    request = StartRequest(
        camera_id="CAM-RTSP-001",
        source_type="rtsp",
        source_location="rtsp://different.local/live",
    )

    with pytest.raises(HTTPException) as exc:
        validate_camera_identity(request)

    assert exc.value.status_code == 422


def test_usb_default_index_matches_registry(isolated_registry):
    request = StartRequest(
        camera_id="CAM-USB-001",
        source_type="usb",
    )

    assert validate_camera_identity(request) == "CAM-USB-001"


def test_legacy_unbound_profile(isolated_registry):
    registry().register(
        camera_id="CAM-LEGACY",
        source_type="usb",
        stream={"width_px": 640, "height_px": 480, "fps": 30},
    )

    request = StartRequest(
        camera_id="CAM-LEGACY",
        source_type="usb",
        source_location=0,
    )

    assert validate_camera_identity(request) == "CAM-LEGACY"
