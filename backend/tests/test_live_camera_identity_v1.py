import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.camera_management import registry
from app.api.live_camera import (
    StartRequest,
    validate_camera_identity,
)


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYGUARD_CAMERA_REGISTRY",
        str(tmp_path / "cameras.json"),
    )
    registry.cache_clear()

    try:
        camera_registry = registry()
        camera_registry.register(
            camera_id="CAM-001",
            source_type="usb",
            stream={"width_px": 640, "height_px": 480},
        )
        yield camera_registry
    finally:
        registry.cache_clear()


def test_legacy_request_without_camera_id():
    request = StartRequest(camera_index=0)

    assert validate_camera_identity(request) is None


def test_registered_camera_accepted(isolated_registry):
    request = StartRequest(
        camera_id="CAM-001",
        source_type="usb",
        source_location=0,
    )

    assert validate_camera_identity(request) == "CAM-001"


def test_unknown_camera_rejected(isolated_registry):
    request = StartRequest(
        camera_id="UNKNOWN",
        source_type="usb",
    )

    with pytest.raises(HTTPException) as exc:
        validate_camera_identity(request)

    assert exc.value.status_code == 404


def test_camera_source_type_mismatch(isolated_registry):
    request = StartRequest(
        camera_id="CAM-001",
        source_type="rtsp",
        source_location="rtsp://camera.example/live",
    )

    with pytest.raises(HTTPException) as exc:
        validate_camera_identity(request)

    assert exc.value.status_code == 422


@pytest.mark.parametrize(
    "camera_id",
    ["", "CAM 001", "../CAM", "CAM/001", "A" * 129],
)
def test_invalid_camera_id_rejected(camera_id):
    with pytest.raises(ValidationError):
        StartRequest(camera_id=camera_id)
