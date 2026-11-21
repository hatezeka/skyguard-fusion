"""39.2D: restart-safe identity and fail-closed live bearing requests."""
from dataclasses import dataclass
from uuid import UUID

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import geo, live_camera
from app.detection.live_session import LiveSessionManager, SessionSnapshot
from app.localization.camera_profile import CameraGeoProfile
from app.localization.live_frame_bearings import LiveBearingRequest, derive_live_frame_bearings
from app.main import app

SESSION_A = "0123456789abcdef0123456789abcdef"
SESSION_B = "fedcba9876543210fedcba9876543210"


def request(**changes):
    values = dict(camera_id="camA", expected_frame_index=42,
                  expected_session_id=SESSION_A, north_reference="true_north",
                  pitch_degrees=0, roll_degrees=0)
    values.update(changes)
    return LiveBearingRequest(**values)


def profile():
    return CameraGeoProfile(camera_id="camA", latitude=42.0,
                            longitude=21.0, heading_degrees=90.0,
                            horizontal_fov_degrees=90.0,
                            coverage_radius_m=1000.0)


def frame_bytes():
    ok, image = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))
    assert ok
    return image.tobytes()


@dataclass
class Snapshot:
    state: str = "running"
    frame_index: int = 42
    camera_id: str | None = "camA"
    session_id: str | None = SESSION_A
    detections: list | None = None


def test_snapshot_backward_compatible_positional_arguments():
    snap = SessionSnapshot("idle", None, [], [], None, None)
    assert snap.session_id is None


def test_distinct_session_ids_after_restart(monkeypatch):
    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def is_alive(self):
            return False

        def start(self):
            pass

    monkeypatch.setattr("app.detection.live_session.threading.Thread", NoopThread)
    manager = LiveSessionManager()
    manager.start(None, None, camera_id="camA")
    first = manager.status()
    manager.start(None, None, camera_id="camA")
    second = manager.status()
    assert first.state == "starting" and second.state == "starting"
    assert first.session_id != second.session_id
    assert UUID(hex=first.session_id).hex == first.session_id
    assert UUID(hex=second.session_id).hex == second.session_id
    assert first.camera_id == second.camera_id == "camA"


def test_matching_session_produces_verified_identity_only():
    payload = derive_live_frame_bearings(
        Snapshot(detections=[]), frame_bytes(), request=request(), profile=profile()
    )
    assert payload["session_id"] == SESSION_A
    assert payload["session_identity_verified"] is True
    assert payload["profile_bound_to_frame_verified"] is False
    assert payload["optics_source"] == "nominal_fov"
    assert payload["physical_distance_available"] is False
    assert payload["target_gps_available"] is False


@pytest.mark.parametrize("active_id", [None, "", SESSION_B])
def test_wrong_or_missing_active_session_rejected(active_id):
    with pytest.raises(ValueError, match="session_id"):
        derive_live_frame_bearings(
            Snapshot(detections=[], session_id=active_id), frame_bytes(),
            request=request(), profile=profile(),
        )


def test_restarted_session_same_camera_same_frame_rejected():
    with pytest.raises(ValueError, match="session_id"):
        derive_live_frame_bearings(
            Snapshot(detections=[], session_id=SESSION_B), frame_bytes(),
            request=request(expected_session_id=SESSION_A), profile=profile()
        )


@pytest.mark.parametrize("bad", [None, "", "short", "x" * 32, True, 7])
def test_missing_or_invalid_session_token_rejected(bad):
    with pytest.raises(ValidationError):
        request(expected_session_id=bad)


def test_session_token_mandatory_in_request_schema():
    data = request().model_dump()
    data.pop("expected_session_id")
    with pytest.raises(ValidationError):
        LiveBearingRequest(**data)


def test_http_bearings_rejects_wrong_session_without_geo_lookup(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    monkeypatch.setattr(live_camera.manager, "latest_frame", lambda: (
        Snapshot(detections=[], session_id=SESSION_B), frame_bytes()
    ))
    def forbidden(*args):
        raise AssertionError("Geo profile must not be read for stale session")
    monkeypatch.setattr(geo, "get_camera_profile", forbidden)
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings", json=request().model_dump(),
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 409


def test_http_bearings_rejects_missing_session_id(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    payload = request().model_dump()
    payload.pop("expected_session_id")
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings", json=payload,
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 422


def test_http_bearings_current_session_succeeds(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    monkeypatch.setattr(live_camera.manager, "latest_frame", lambda: (
        Snapshot(detections=[]), frame_bytes()
    ))
    monkeypatch.setattr(geo, "get_camera_profile", lambda camera_id: {
        "camera_id": camera_id, "latitude": 42.0, "longitude": 21.0,
        "heading_degrees": 90.0, "horizontal_fov_degrees": 90.0,
        "coverage_radius_m": 1000.0,
    })
    response = TestClient(app).post(
        "/api/v1/live/frame/bearings", json=request().model_dump(),
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["session_id"] == SESSION_A
    assert response.json()["session_identity_verified"] is True


def test_http_frame_exposes_identity_from_single_snapshot(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    snapshot = SessionSnapshot("running", 42, [], [], 0.01, None,
                               camera_id="camA", session_id=SESSION_A)
    monkeypatch.setattr(live_camera.manager, "latest_frame", lambda: (snapshot, b"jpeg-data"))
    response = TestClient(app).get(
        "/api/v1/live/frame", headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["camera_id"] == "camA"
    assert response.json()["session_id"] == SESSION_A


def test_http_detections_exposes_identity_from_status(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    snapshot = SessionSnapshot("running", 42, [], [], 0.01, None,
                               camera_id="camA", session_id=SESSION_A)
    monkeypatch.setattr(live_camera.manager, "status", lambda: snapshot)
    response = TestClient(app).get(
        "/api/v1/live/detections", headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["camera_id"] == "camA"
    assert response.json()["session_id"] == SESSION_A
