"""Synthetic coordinate and shared-event tests. No physical claims."""
from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError

from app.localization.rig_reference import RigReference, camera_xyz_to_enu, project_verified_position
from app.phase43_45.maps import live_localization_scene


@pytest.mark.parametrize(("heading", "expected"), [
    (0, (0, 20, 0)), (90, (20, 0, 0)), (180, (0, -20, 0)), (270, (-20, 0, 0)),
])
def test_synthetic_cardinal_axes(heading, expected):
    assert camera_xyz_to_enu((0, 0, 20), heading, 0, 0) == pytest.approx(expected)


def test_synthetic_projection_wrap_and_invalid_references():
    now = datetime.now(timezone.utc)
    ref = RigReference(latitude=0, longitude=179.99995, altitude_m=20, altitude_datum="ellipsoidal",
                       source="gnss", heading_deg=90, pitch_deg=0, roll_deg=0,
                       antenna_offset_xyz_m=(0, 0, 5), captured_at=now,
                       valid_until=now + timedelta(hours=1), evidence_ref="synthetic-reference")
    result = project_verified_position((0, 0, 25), ref, stereo_valid=True)
    assert result["latitude"] == pytest.approx(0)
    assert -180 < result["longitude"] < -179.999
    assert result["enu_m"] == pytest.approx([20, 0, 0])
    assert result["uncertainty_status"] == "UNCERTAINTY_UNAVAILABLE"
    assert project_verified_position((0, 0, 25), ref, stereo_valid=False) is None
    assert project_verified_position((0, 0, -25), ref, stereo_valid=True) is None
    assert ref.model_copy(update={"altitude_datum": "orthometric"}).status() == "GEO_REFERENCE_INCOMPLETE"
    assert ref.status(now + timedelta(hours=2)) == "GEO_REFERENCE_STALE"
    with pytest.raises(ValidationError):
        RigReference.model_validate({**ref.model_dump(), "latitude": float("inf")})


def test_both_maps_share_the_same_unavailable_observation():
    scene = live_localization_scene("session", 42, "2026-11-22T12:00:00Z", [
        {"classification": "DRONE", "track_id": 8, "range_status": "CALIBRATION_SUSPECT"},
        {"classification": "UNKNOWN", "track_id": 9, "range_status": "RANGE_UNAVAILABLE"},
    ])
    item, = scene["objects"]
    assert item["measurement_id"] == "session:42:8"
    assert item["timestamp_utc"] == scene["timestamp_utc"]
    assert item["geographic"]["latitude"] is None and item["localization"]["xyz_m"] is None


def test_local_rig_reference_api_synthetic(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.skyguard_persistence.live_bridge import mount_live_bridge

    monkeypatch.setenv("SKYGUARD_RIG_REFERENCE", str(tmp_path / "reference.json"))
    monkeypatch.setenv("SKYGUARD_DB_API_KEY", "synthetic-test-token")
    app = FastAPI()
    mount_live_bridge(app, None)
    # Exercise only the file-backed reference endpoints; no database lifespan.
    client = TestClient(app)
    endpoint = "/api/persistence/live/rig-reference"
    auth = {"Authorization": "Bearer synthetic-test-token"}
    assert client.get(endpoint).status_code == 401
    assert client.get(endpoint, headers=auth).json()["status"] == "GEO_REFERENCE_MISSING"
    payload = {"latitude": 10, "longitude": 20, "source": "manual",
               "captured_at": "2026-11-22T06:00:00Z"}
    response = client.post(endpoint, headers=auth, json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "GEO_REFERENCE_INCOMPLETE"
    assert client.get(endpoint, headers=auth).json() == response.json()
    assert client.post(endpoint, headers=auth, json={**payload, "latitude": 91}).status_code == 422


def test_synthetic_uncertainty_propagation_requires_known_accuracy_convention():
    now = datetime.now(timezone.utc)
    ref = RigReference(latitude=0, longitude=0, altitude_m=2, altitude_datum="ellipsoidal",
                       source="gnss", heading_deg=0, pitch_deg=0, roll_deg=0,
                       antenna_offset_xyz_m=(0,0,0), captured_at=now,
                       valid_until=now+timedelta(hours=1), evidence_ref="synthetic",
                       heading_uncertainty_deg=0, pitch_uncertainty_deg=0, roll_uncertainty_deg=0,
                       horizontal_accuracy_m=1, vertical_accuracy_m=2)
    kwargs = {"stereo_valid": True, "covariance_xyz_m2": [[1,0,0],[0,1,0],[0,0,1]],
              "transform_sigma_m": 0}
    assert project_verified_position((0,0,10), ref, **kwargs)["uncertainty_status"] == "UNCERTAINTY_UNAVAILABLE"
    result = project_verified_position((0,0,10), ref, gps_accuracy_is_one_sigma=True, **kwargs)
    assert result["horizontal_rms_m"] == pytest.approx(2)
    assert result["vertical_sigma_m"] == pytest.approx(5**.5)
