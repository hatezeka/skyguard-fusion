import pytest

from app.camera_management.core import CameraRegistry, CameraError


def test_usb_camera_source_binding(tmp_path):
    registry = CameraRegistry(tmp_path / "cameras.json")

    profile = registry.register(
        camera_id="CAM-USB-001",
        source_type="usb",
        stream={
            "width_px": 640,
            "height_px": 480,
            "fps": 30,
        },
        source_location=0,
    )

    assert profile["source_location"] == 0

    reloaded = CameraRegistry(tmp_path / "cameras.json")
    assert reloaded.get("CAM-USB-001")["source_location"] == 0


def test_rtsp_camera_source_binding(tmp_path):
    registry = CameraRegistry(tmp_path / "cameras.json")

    profile = registry.register(
        camera_id="CAM-RTSP-001",
        source_type="rtsp",
        stream={
            "width_px": 1920,
            "height_px": 1080,
            "fps": 25,
        },
        source_location="rtsp://example.local/live",
    )

    assert profile["source_location"] == "rtsp://example.local/live"


@pytest.mark.parametrize(
    "source_type,source_location",
    [
        ("usb", -1),
        ("usb", True),
        ("usb", "0"),
        ("rtsp", 0),
        ("rtsp", "http://example.local/live"),
        ("rtsp", "rtsp://"),
    ],
)
def test_invalid_camera_source_binding(
    tmp_path,
    source_type,
    source_location,
):
    registry = CameraRegistry(tmp_path / "cameras.json")

    with pytest.raises(CameraError):
        registry.register(
            camera_id="CAM-INVALID",
            source_type=source_type,
            stream={
                "width_px": 640,
                "height_px": 480,
                "fps": 30,
            },
            source_location=source_location,
        )


def test_legacy_camera_profile_remains_readable(tmp_path):
    registry = CameraRegistry(tmp_path / "cameras.json")

    registry.register(
        camera_id="CAM-LEGACY",
        source_type="usb",
        stream={
            "width_px": 640,
            "height_px": 480,
            "fps": 30,
        },
    )

    profile = registry.get("CAM-LEGACY")

    assert profile.get("source_location") is None
