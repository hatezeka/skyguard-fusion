"""Phase 39.2C tests: atomic live image/detection direction API."""
from dataclasses import dataclass

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import geo, live_camera
from app.main import app
from app.localization.camera_profile import CameraGeoProfile
from app.localization.live_frame_bearings import LiveBearingRequest, derive_live_frame_bearings


@dataclass
class FakeSnapshot:
    state: str = "running"
    frame_index: int = 42
    camera_id: str | None = "camA"
    session_id: str | None = "0123456789abcdef0123456789abcdef"
    detections: list | None = None


def image_bytes(width=1280, height=720):
    success, encoded = cv2.imencode(
        ".jpg", np.zeros((height, width, 3), dtype=np.uint8)
    )
    assert success
    return encoded.tobytes()


def detection(label="drone", index=None):
    value = {
        "class": label,
        "confidence": 0.9,
        "bbox": {"x": 630, "y": 350, "width": 20, "height": 20},
    }
    if index is not None:
        value["frame_index"] = index
    return value


def profile(camera_id="camA"):
    return CameraGeoProfile(
        camera_id=camera_id, latitude=42.0, longitude=21.0,
        heading_degrees=90.0, horizontal_fov_degrees=90.0,
        coverage_radius_m=1000.0,
    )


def request(**changes):
    data = {
        "camera_id": "camA", "expected_frame_index": 42,
        "expected_session_id": "0123456789abcdef0123456789abcdef",
        "north_reference": "true_north", "pitch_degrees": 0,
        "roll_degrees": 0,
    }
    data.update(changes)
    return LiveBearingRequest(**data)


def test_atomic_bridge_center_bearing_without_target_gps():
    payload = derive_live_frame_bearings(
        FakeSnapshot(detections=[detection()]), image_bytes(),
        request=request(), profile=profile(),
    )
    assert payload["frame_index"] == 42
    assert payload["image_width_px"] == 1280
    assert payload["image_height_px"] == 720
    assert payload["directions"][0]["direction"]["azimuth_degrees"] == pytest.approx(90)
    assert payload["optics_source"] == "nominal_fov"
    assert payload["orientation_source"] == "manual_unverified"
    assert payload["session_identity_verified"] is True
    assert payload["profile_bound_to_frame_verified"] is False
    assert payload["target_gps_available"] is False
    assert "latitude" not in payload["directions"][0]
    assert "distance_m" not in payload["directions"][0]


def test_bridge_uses_actual_jpeg_dimensions_not_assumed_resolution():
    output = derive_live_frame_bearings(
        FakeSnapshot(detections=[]), image_bytes(640, 480),
        request=request(), profile=profile(),
    )
    assert (output["image_width_px"], output["image_height_px"]) == (640, 480)
    assert output["directions"] == []


@pytest.mark.parametrize("state", ["idle", "starting", "stopped", "error"])
def test_bridge_rejects_nonrunning_frame(state):
    with pytest.raises(ValueError, match="not running"):
        derive_live_frame_bearings(
            FakeSnapshot(state=state, detections=[]), image_bytes(),
            request=request(), profile=profile(),
        )


def test_bridge_rejects_stale_index():
    with pytest.raises(ValueError, match="frame_index"):
        derive_live_frame_bearings(
            FakeSnapshot(detections=[]), image_bytes(),
            request=request(expected_frame_index=41), profile=profile(),
        )


def test_bridge_rejects_mismatched_camera():
    with pytest.raises(ValueError, match="camera_id"):
        derive_live_frame_bearings(
            FakeSnapshot(detections=[]), image_bytes(),
            request=request(camera_id="camB"), profile=profile(),
        )


def test_bridge_rejects_missing_camera_identity():
    with pytest.raises(ValueError, match="camera_id"):
        derive_live_frame_bearings(
            FakeSnapshot(camera_id=None, detections=[]), image_bytes(),
            request=request(), profile=profile(),
        )


def test_bridge_rejects_stale_detection():
    with pytest.raises(ValueError, match="frame_index"):
        derive_live_frame_bearings(
            FakeSnapshot(detections=[detection(index=41)]), image_bytes(),
            request=request(), profile=profile(),
        )


def test_bridge_rejects_corrupt_jpeg():
    with pytest.raises(ValueError, match="JPEG"):
        derive_live_frame_bearings(
            FakeSnapshot(detections=[]), b"not-jpeg",
            request=request(), profile=profile(),
        )


def test_bridge_rejects_mismatched_geo_profile():
    with pytest.raises(ValueError, match="Geo profile"):
        derive_live_frame_bearings(
            FakeSnapshot(detections=[]), image_bytes(),
            request=request(), profile=profile("other"),
        )


def test_bridge_unknown_stays_unknown():
    result = derive_live_frame_bearings(
        FakeSnapshot(detections=[detection("unknown")]), image_bytes(),
        request=request(), profile=profile(),
    )
    assert result["directions"][0]["object_class"] == "unknown"


def test_http_route_bearings_success_and_atomic_read(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    calls = {"frame": 0}

    def atomic_pair():
        calls["frame"] += 1
        return FakeSnapshot(detections=[detection()]), image_bytes()

    monkeypatch.setattr(live_camera.manager, "latest_frame", atomic_pair)
    monkeypatch.setattr(geo, "get_camera_profile", lambda cid: {
        "camera_id": cid, "latitude": 42.0, "longitude": 21.0,
        "heading_degrees": 90.0, "horizontal_fov_degrees": 90.0,
        "coverage_radius_m": 1000.0,
    })
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings",
        json=request().model_dump(),
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200, response.text
    assert calls["frame"] == 1
    assert response.json()["directions"][0]["direction"]["azimuth_degrees"] == pytest.approx(90)


def test_http_route_rejects_stale_frame(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    monkeypatch.setattr(live_camera.manager, "latest_frame", lambda: (
        FakeSnapshot(detections=[]), image_bytes(),
    ))
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings",
        json=request(expected_frame_index=41).model_dump(),
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 409


def test_http_route_rejects_unauthenticated(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings",
        json=request().model_dump(),
    )
    assert response.status_code in (401, 403)
