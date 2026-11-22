from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.localization.session_calibration import freeze_session_calibration
from app.skyguard_persistence.live_bridge import _frame_geometry


@pytest.fixture
def frame():
    snapshot = SimpleNamespace(camera_id="usb-test", session_id="a" * 32)
    ok, data = cv2.imencode(".jpg", np.zeros((100, 200), dtype=np.uint8))
    assert ok
    record = {
        "camera_id": snapshot.camera_id,
        "stream": {"width_px": 200, "height_px": 100},
        "calibration": {
            "status": "CALIBRATED", "resolution": [200, 100], "verified_at": "fixture",
            "intrinsics": {"fx": 100, "fy": 100, "cx": 100, "cy": 50, "distortion": [0] * 5},
        },
    }
    return snapshot, data.tobytes(), record


def test_bound_jpeg_yields_relative_direction_without_position(frame):
    snapshot, jpeg, record = frame
    detections = [{"bbox": [110, 39, 130, 41], "track_id": None, "classification": "UNKNOWN"}]
    binding = freeze_session_calibration(record, snapshot.camera_id, snapshot.session_id)
    info = _frame_geometry(snapshot, jpeg, binding, detections)
    assert info is not None and info["physical_calibration_verified"] is False
    assert info["horizontal_fov_deg"] == pytest.approx(89.7, abs=0.5)
    detection, = detections
    assert detection["relative_bearing_deg"] > 0 and detection["relative_elevation_deg"] > 0
    assert detection["track_id"] is None
    assert "azimuth_deg" not in detection and "range_m" not in detection and "latitude" not in detection


def test_rejects_mismatched_session_and_uses_explicit_north_reference(frame):
    snapshot, jpeg, record = frame
    detections = [{"bbox": [99, 49, 101, 51]}]
    mismatched = freeze_session_calibration(record, snapshot.camera_id, "b" * 32)
    assert _frame_geometry(snapshot, jpeg, mismatched, detections) is None
    record["orientation"] = {
        "source": "calibration", "north_reference": "true_north", "evidence_ref": "fixture",
        "yaw_deg": 90, "pitch_deg": 0, "roll_deg": 0,
    }
    binding = freeze_session_calibration(record, snapshot.camera_id, snapshot.session_id)
    assert _frame_geometry(snapshot, jpeg, binding, detections) is not None
    assert detections[0]["azimuth_deg"] == pytest.approx(90)
