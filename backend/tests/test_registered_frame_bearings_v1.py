"""Phase 39.2E: registry optical data applied to correct live session/frame."""

from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import live_camera, camera_management
from app.localization.registered_frame_bearings import (
    RegisteredBearingRequest,
    derive_registered_frame_bearings,
)
from app.main import app

SESSION = "0123456789abcdef0123456789abcdef"
OTHER_SESSION = "fedcba9876543210fedcba9876543210"


@dataclass
class Frame:
    state: str = "running"
    camera_id: str | None = "camA"
    frame_index: int = 42
    session_id: str | None = SESSION
    detections: list | None = None


def jpeg(width=200, height=100):
    success, encoded = cv2.imencode(
        ".jpg", np.zeros((height, width, 3), dtype=np.uint8)
    )
    assert success
    return encoded.tobytes()


def registry():
    # Does not include geographical coordinates; the ray must still work.
    return {
        "camera_id": "camA",
        "stream": {"width_px": 200, "height_px": 100, "fps": 25.0},
        "geolocation": {"latitude": None, "longitude": None, "source": "unavailable"},
        "calibration": {
            "status": "CALIBRATED",
            "verified_at": "2026-11-20T14:00:00Z",
            "resolution": [200, 100],
            "intrinsics": {
                "fx": 100.0, "fy": 100.0, "cx": 100.0, "cy": 50.0,
                "distortion": [0, 0, 0, 0, 0],
            },
        },
        "orientation": {
            "yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0,
            "source": "manual",
        },
    }


def request(**changes):
    args = {
        "camera_id": "camA", "expected_frame_index": 42,
        "expected_session_id": SESSION, "north_reference": "true_north",
    }
    args.update(changes)
    return RegisteredBearingRequest(**args)


def detect(x=90, y=40, w=20, h=20, klass="drone"):
    return {
        "class": klass, "confidence": 0.9,
        "bbox": {"x": x, "y": y, "width": w, "height": h},
    }


def derive(record=None, snapshot=None, jpeg_bytes=None, req=None):
    return derive_registered_frame_bearings(
        snapshot if snapshot is not None else Frame(detections=[detect()]),
        jpeg_bytes if jpeg_bytes is not None else jpeg(),
        registry_record=record if record is not None else registry(),
        request=req if req is not None else request(),
    )


def test_center_ray_uses_registered_camera_not_fabricated_gps():
    result = derive()
    assert result["directions"][0]["direction"]["azimuth_degrees"] == pytest.approx(90)
    assert result["frame_source"] == "manager.latest_frame_atomic"
    assert result["session_id"] == SESSION
    assert result["optics_source"] == "registry_intrinsics_unverified"
    assert result["orientation_source"] == "registry_manual_unverified"
    assert result["session_identity_verified"] is True
    assert result["profile_bound_to_session_verified"] is False
    assert result["profile_bound_to_frame_verified"] is False
    assert result["physical_calibration_verified"] is False
    assert result["physical_distance_available"] is False
    assert result["target_gps_available"] is False
    assert "latitude" not in str(result)
    assert "distance_m" not in str(result)


def test_uses_lens_distortion_instead_of_nominal_fov():
    record = registry()
    record["calibration"]["intrinsics"]["distortion"][0] = 0.7
    # The bbox center is (155, 50), so distortion correction is nontrivial.
    data = derive(record=record, snapshot=Frame(detections=[detect(x=145, y=40)]))
    found = data["directions"][0]["direction"]
    matrix = np.array([[100, 0, 100], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
    expected = cv2.undistortPoints(
        np.array([[[155., 50.]]], dtype=np.float64),
        matrix,
        np.array([0.7, 0, 0, 0, 0], dtype=np.float64),
    )[0, 0]
    assert found["azimuth_degrees"] == pytest.approx(
        (90 + np.degrees(np.arctan(expected[0]))) % 360, abs=1e-5
    )
    assert found["optics_source"] == "unverified"


def test_registry_pitch_changes_elevation():
    r = registry()
    r["orientation"]["pitch_deg"] = 18.0
    data = derive(record=r)
    assert data["directions"][0]["direction"]["elevation_degrees"] == pytest.approx(18)


def test_unknown_remains_unknown():
    value = derive(snapshot=Frame(detections=[detect(klass="unknown")]))
    assert value["directions"][0]["object_class"] == "unknown"


def test_empty_detections_yields_empty_directions():
    assert derive(snapshot=Frame(detections=[]))["directions"] == []


@pytest.mark.parametrize("state", ["stopped", "idle", "error", "starting"])
def test_not_running_refused(state):
    with pytest.raises(ValueError, match="not running"):
        derive(snapshot=Frame(state=state))


@pytest.mark.parametrize("changes", [
    {"camera_id": "other"},
    {"frame_index": 43},
    {"session_id": OTHER_SESSION},
    {"session_id": None},
])
def test_stale_frame_or_camera_or_session_refused(changes):
    with pytest.raises(ValueError, match="camera_id|frame_index|session_id"):
        derive(snapshot=Frame(detections=[], **changes))


@pytest.mark.parametrize("change,expected", [
    ({"camera_id": "wrong"}, "camera_id"),
    ({"stream": {"width_px": 201, "height_px": 100}}, "resolution"),
    ({"calibration": {"status": "INVALIDATED"}}, "not CALIBRATED"),
    ({"calibration": {"status": "UNCALIBRATED"}}, "not CALIBRATED"),
    ({"orientation": {"source": "unavailable"}}, "orientation"),
])
def test_registry_contract_mismatch_refused(change, expected):
    record = registry()
    for name, values in change.items():
        if isinstance(values, dict) and name in record and isinstance(record[name], dict):
            record[name].update(values)
        else:
            record[name] = values
    with pytest.raises(ValueError, match=expected):
        derive(record=record)


def test_invalid_intrinsics_nan_refused():
    record = registry()
    record["calibration"]["intrinsics"]["fx"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        derive(record=record)


def test_invalid_distortion_count_refused():
    record = registry()
    record["calibration"]["intrinsics"]["distortion"] = [0, 0]
    with pytest.raises(ValueError, match="Five registry"):
        derive(record=record)


def test_calibration_resolution_invalidation_refused():
    record = registry()
    record["calibration"]["resolution"] = [200, 101]
    with pytest.raises(ValueError, match="resolution"):
        derive(record=record)


def test_calibration_timestamp_missing_refused():
    record = registry()
    record["calibration"]["verified_at"] = None
    with pytest.raises(ValueError, match="timestamp"):
        derive(record=record)


def test_actual_jpeg_resolution_refused():
    with pytest.raises(ValueError, match="resolution"):
        derive(jpeg_bytes=jpeg(250, 100))


def test_bbox_outside_image_refused():
    with pytest.raises(ValueError, match="outside image"):
        derive(snapshot=Frame(detections=[detect(x=190, y=40, w=20)]))


@pytest.mark.parametrize("bad", [None, "short", "x" * 32, True])
def test_invalid_session_request_rejected(bad):
    with pytest.raises(ValidationError):
        request(expected_session_id=bad)


def test_missing_north_reference_rejected():
    payload = request().model_dump()
    payload.pop("north_reference")
    with pytest.raises(ValidationError):
        RegisteredBearingRequest(**payload)


class FakeRegistry:
    def __init__(self, record=None):
        self.record = record if record is not None else registry()
        self.calls = 0

    def get(self, camera_id):
        self.calls += 1
        if camera_id != "camA":
            from app.camera_management.core import CameraError
            raise CameraError("unknown camera")
        return deepcopy(self.record)


def test_http_success_once_from_snapshot_once_from_registry(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    counts = {"frame": 0}

    def latest_frame():
        counts["frame"] += 1
        return Frame(detections=[detect()]), jpeg()

    profile = FakeRegistry()
    monkeypatch.setattr(live_camera.manager, "latest_frame", latest_frame)
    monkeypatch.setattr(camera_management, "registry", lambda: profile)
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings/registered",
        json=request().model_dump(),
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200, response.text
    assert counts["frame"] == 1
    assert profile.calls == 1
    assert response.json()["directions"][0]["direction"]["azimuth_degrees"] == pytest.approx(90)


def test_http_wrong_session_refused_before_registry_access(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    fake = FakeRegistry()
    monkeypatch.setattr(camera_management, "registry", lambda: fake)
    monkeypatch.setattr(live_camera.manager, "latest_frame", lambda: (
        Frame(detections=[], session_id=OTHER_SESSION), jpeg()
    ))
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings/registered",
        json=request().model_dump(),
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 409
    assert fake.calls == 0


def test_http_authentication_required(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings/registered", json=request().model_dump()
    )
    assert response.status_code in (401, 403)


def test_http_bad_registry_calibration_returns_409(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    invalid = registry()
    invalid["calibration"]["status"] = "INVALIDATED"
    monkeypatch.setattr(camera_management, "registry", lambda: FakeRegistry(invalid))
    monkeypatch.setattr(live_camera.manager, "latest_frame", lambda: (
        Frame(detections=[]), jpeg()
    ))
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings/registered",
        json=request().model_dump(),
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 409


def test_http_missing_session_id_returns_422(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    payload = request().model_dump()
    payload.pop("expected_session_id")
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings/registered", json=payload,
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 422
