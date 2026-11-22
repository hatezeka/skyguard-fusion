import cv2
import numpy as np
import pytest

from app.camera_management.core import CameraError, CameraRegistry
from app.video.source import UniversalVideoSource, VideoSourceConfig


@pytest.fixture
def registry(tmp_path):
    result = CameraRegistry(tmp_path / "cameras.json")
    result.register("usb", "usb", {"width_px": 640, "height_px": 480, "fps": 30}, 1)
    return result


def candidate():
    return {
        "image_size_px": [640, 480],
        "camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        "distortion_coefficients": [0] * 5,
        "global_reprojection_rms_px": 1.2,
    }


def test_import_versions_only_changed_optics(registry):
    first = registry.import_usb_calibration("usb", candidate(), "Fixed focus")["calibration"]
    repeated = registry.import_usb_calibration("usb", candidate(), "Fixed focus")["calibration"]
    changed = registry.import_usb_calibration("usb", candidate(), "Changed optics")
    assert first == repeated
    assert first["validation_status"] == "CALIBRATED_UNVALIDATED"
    assert len(changed["calibration_profiles"]) == 2


@pytest.mark.parametrize("focal_length", [0, -20, float("nan"), float("inf")])
def test_invalid_focal_length_leaves_camera_uncalibrated(registry, focal_length):
    profile = candidate()
    profile["camera_matrix"][0][0] = focal_length
    with pytest.raises(CameraError):
        registry.import_usb_calibration("usb", profile, "Fixed focus")
    assert registry.get("usb")["calibration"]["status"] == "UNCALIBRATED"


class Capture:
    def __init__(self, images=(), opened=True):
        self.images = iter(images)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, _property, _value):
        return True

    def get(self, _property):
        return cv2.VideoWriter_fourcc(*"MJPG")

    def read(self):
        return True, next(self.images)

    def release(self):
        self.released = True


def test_black_frame_and_exposure_warmup(monkeypatch):
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    bright = np.full_like(black, 35)
    handle = Capture([black, None, bright, bright, bright])
    calls = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: calls.append(args) or handle)
    with UniversalVideoSource(VideoSourceConfig("usb", 1, (640, 480))) as source:
        frame = source.read()
    assert calls == [(1, cv2.CAP_DSHOW)]
    assert handle.released and frame.index == 0
    assert np.array_equal(frame.image, bright)


def test_wrong_actual_resolution_is_rejected(monkeypatch):
    handle = Capture([np.full((720, 1280, 3), 35, dtype=np.uint8)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: handle)
    with UniversalVideoSource(VideoSourceConfig("usb", 1, (640, 480))) as source:
        with pytest.raises(RuntimeError, match="resolution mismatch"):
            source.read()
    assert handle.released


def test_missing_device_does_not_fall_back_to_another_index(monkeypatch):
    handle = Capture(opened=False)
    calls = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: calls.append(args) or handle)
    with pytest.raises(RuntimeError, match="USB camera unavailable"):
        UniversalVideoSource(VideoSourceConfig("usb", 1, (640, 480))).open()
    assert calls == [(1, cv2.CAP_DSHOW)]
    assert handle.released
