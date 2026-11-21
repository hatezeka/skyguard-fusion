"""Phase 40.1: trustworthy live-map camera + display-only bearing geometry."""
from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import live_camera
from app.detection.live_session import SessionSnapshot
from app.localization.live_map_scene import LiveMapSceneRequest, build_live_map_scene
from app.localization.session_calibration import freeze_session_calibration
from app.main import app

SESSION = "0123456789abcdef0123456789abcdef"
OTHER = "fedcba9876543210fedcba9876543210"
URL = "/api/v1/live/map/scene"


def camera_record(*, lat=42.66, lon=21.16, source="manual", yaw=90.0):
    return {
        "camera_id": "camA",
        "source_location": "rtsp://hidden-user:hidden-pass@host/camera?token=never-expose",
        "private_key": "never-expose",
        "stream": {"width_px": 200, "height_px": 100},
        "calibration": {
            "status": "CALIBRATED", "resolution": [200, 100],
            "verified_at": "2026-11-20T14:00:00Z",
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 100.0, "cy": 50.0,
                           "distortion": [0.0] * 5},
        },
        "orientation": {"yaw_deg": yaw, "pitch_deg": 0.0, "roll_deg": 0.0, "source": "manual"},
        "geolocation": {
            "latitude": lat, "longitude": lon, "source": source,
            "horizontal_accuracy_m": None, "captured_at": None,
            "source_location": "SECRET", "password": "SECRET",
        },
    }


def bearing_data(*, azimuth=90.0, north_reference="true_north", detections=None):
    if detections is None:
        detections = [{
            "detection_index": 0, "object_class": "unknown", "confidence": 0.75,
            "direction": {"azimuth_degrees": azimuth,
                          "elevation_degrees": 5.0,
                          "north_reference": north_reference},
        }]
    return {
        "camera_id": "camA", "session_id": SESSION, "frame_index": 7,
        "north_reference": north_reference, "directions": detections,
    }


def scene(record=None, bearings=None, reference="true_north"):
    return build_live_map_scene(
        bearing_data(north_reference=reference) if bearings is None else bearings,
        camera_record() if record is None else record,
        calibration_sha256="a" * 64,
        north_reference=reference,
        expected_camera_id="camA", expected_session_id=SESSION, expected_frame_index=7,
    )


def snapshot(*, camera="camA", session=SESSION, frame=7, state="running"):
    return SessionSnapshot(
        state, frame,
        [{"class": "drone", "confidence": .9,
          "bbox": {"x": 90, "y": 40, "width": 20, "height": 20}}],
        [], .01, None, camera_id=camera, session_id=session,
    )


def jpeg():
    ok, data = cv2.imencode(".jpg", np.zeros((100, 200, 3), dtype=np.uint8))
    assert ok
    return data.tobytes()


def request(**updates):
    result = {"camera_id": "camA", "expected_frame_index": 7,
              "expected_session_id": SESSION, "north_reference": "true_north"}
    result.update(updates)
    return result


def test_scene_has_geojson_camera_and_guide():
    result = scene()
    assert result["geojson"]["type"] == "FeatureCollection"
    a, b = result["geojson"]["features"]
    assert a["properties"]["kind"] == "camera"
    assert a["geometry"]["coordinates"] == [21.16, 42.66]
    assert b["properties"]["kind"] == "bearing_guide"
    assert b["properties"]["display_only"] is True
    assert b["properties"]["is_target_position"] is False
    assert b["properties"]["guide_length_m"] == 200
    assert b["geometry"]["coordinates"][1][0] > 21.16  # east heading


def test_north_bearing_points_north():
    result = scene(bearings=bearing_data(azimuth=0))
    end = result["geojson"]["features"][1]["geometry"]["coordinates"][1]
    assert end[1] > 42.66
    assert end[0] == pytest.approx(21.16, abs=1e-6)


def test_all_points_are_display_only_not_target_gps():
    result = scene()
    assert result["target_gps_available"] is False
    assert result["target_range_available"] is False
    assert result["bearing_guides_are_display_only"] is True
    assert result["observations"][0]["target_location"] is None
    assert result["observations"][0]["target_range_m"] is None
    assert result["physical_calibration_verified"] is False
    assert result["camera_location_bound_to_session_verified"] is False


def test_magnetic_bearing_does_not_draw_geographic_guide():
    result = scene(
        bearings=bearing_data(north_reference="magnetic_north"),
        reference="magnetic_north",
    )
    assert len(result["geojson"]["features"]) == 1
    assert result["guides_suppressed_reason"] == "magnetic_declination_unavailable"
    assert result["magnetic_declination_applied"] is False


def test_camera_position_missing_does_not_invent_coordinates():
    record = camera_record(lat=None, lon=None, source="unavailable")
    result = scene(record=record)
    assert result["camera_location"] is None
    assert result["geojson"]["features"] == []
    assert result["guides_suppressed_reason"] == "camera_location_unavailable"
    assert result["observations"][0]["azimuth_degrees"] == 90


def test_simulated_location_explicit():
    result = scene(record=camera_record(source="simulation"))
    assert result["camera_location_source"] == "simulation"
    assert result["camera_location_survey_verified"] is False


def test_empty_detections_still_camera_marker():
    result = scene(bearings=bearing_data(detections=[]))
    assert result["observations"] == []
    assert len(result["geojson"]["features"]) == 1


def test_vertical_direction_has_no_guide():
    result = scene(bearings=bearing_data(azimuth=None))
    assert result["observations"][0]["azimuth_degrees"] is None
    assert len(result["geojson"]["features"]) == 1


def test_privacy_no_source_url_secrets():
    record = camera_record()
    record["geolocation"]["credential"] = "private-credential"
    result = str(scene(record=record))
    assert "never-expose" not in result and "SECRET" not in result
    assert "private-credential" not in result and "rtsp" not in result


@pytest.mark.parametrize("field,value", [
    ("latitude", 95), ("longitude", -181),
    ("latitude", float("nan")), ("longitude", True),
    ("latitude", "42.66"), ("horizontal_accuracy_m", -2),
])
def test_invalid_location_fails_closed(field, value):
    record = camera_record()
    record["geolocation"][field] = value
    with pytest.raises(ValueError):
        scene(record=record)


def test_registry_wrong_camera_fails_closed():
    with pytest.raises(ValueError, match="identity"):
        scene(record={**camera_record(), "camera_id": "evil"})


def test_unavailable_source_with_fabricated_coords_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        scene(record=camera_record(source="unavailable"))


def test_camera_location_partial_coordinate_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        scene(record=camera_record(lat=None, lon=21.16))


def test_extreme_latitude_suppresses_guide():
    result = scene(record=camera_record(lat=90, lon=21))
    assert len(result["geojson"]["features"]) == 1


def test_antimeridian_suppresses_globe_spanning_guide():
    result = scene(record=camera_record(lat=0, lon=179.9995), bearings=bearing_data(azimuth=90))
    assert len(result["geojson"]["features"]) == 1


def test_duplicate_detection_index_rejected():
    first = bearing_data()["directions"][0]
    with pytest.raises(ValueError, match="duplicate"):
        scene(bearings=bearing_data(detections=[first, deepcopy(first)]))


def test_detection_bbox_not_duplicated_and_no_track_id_inferred():
    result = scene()
    assert set(result["observations"][0]) == {
        "detection_index", "object_class", "classification_is_confirmed",
        "confidence", "azimuth_degrees", "elevation_degrees",
        "north_reference", "target_location", "target_range_m", "target_location_observable",
    }
    assert result["observations"][0]["classification_is_confirmed"] is False


@pytest.mark.parametrize("change", [
    {"camera_id": "b"}, {"session_id": OTHER}, {"frame_index": -1},
    {"north_reference": "magnetic_north"},
])
def test_inconsistent_frame_provenance_rejected(change):
    data = bearing_data()
    data.update(change)
    with pytest.raises(ValueError):
        scene(bearings=data)


def test_invalid_calibration_hash_rejected():
    with pytest.raises(ValueError):
        build_live_map_scene(bearing_data(), camera_record(),
                             calibration_sha256="bad", north_reference="true_north",
                             expected_camera_id="camA", expected_session_id=SESSION, expected_frame_index=7)


def test_request_rejects_boolean_index():
    with pytest.raises(Exception):
        LiveMapSceneRequest(**request(expected_frame_index=True))


def test_request_rejects_bad_session():
    with pytest.raises(Exception):
        LiveMapSceneRequest(**request(expected_session_id="old"))


def test_endpoint_is_effectively_registered_openapi():
    assert "post" in app.openapi()["paths"][URL]


def test_http_requires_auth(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "phase40-temp-key")
    response = TestClient(app).post(URL, json=request())
    assert response.status_code in (401, 403)


def setup_live(monkeypatch, *, snap=None, frozen=None, profile=None):
    monkeypatch.setenv("SKYGUARD_API_KEY", "phase40-temp-key")
    record = camera_record() if profile is None else profile
    frozen = freeze_session_calibration(record, "camA", SESSION) if frozen is None else frozen
    monkeypatch.setattr(live_camera.manager, "latest_bound_frame", lambda: (
        snapshot() if snap is None else snap, jpeg(), frozen,
    ))
    monkeypatch.setattr(live_camera, "registry", lambda: type("Registry", (), {
        "get": lambda self, camera_id: deepcopy(record),
    })())


def test_http_live_scene_happy_path(monkeypatch):
    setup_live(monkeypatch)
    response = TestClient(app).post(URL, json=request(), headers={"X-API-Key": "phase40-temp-key"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["session_id"] == SESSION
    assert data["frame_index"] == 7
    assert data["calibration_snapshot_sha256"]
    assert len(data["geojson"]["features"]) == 2
    assert data["observations"][0]["target_location"] is None


@pytest.mark.parametrize("change", [
    {"camera": "wrong"}, {"frame": 6}, {"session": OTHER}, {"state": "stopped"},
])
def test_http_stale_camera_frame_or_session_fails_closed(monkeypatch, change):
    setup_live(monkeypatch, snap=snapshot(**change))
    response = TestClient(app).post(URL, json=request(), headers={"X-API-Key": "phase40-temp-key"})
    assert response.status_code == 409


def test_http_rejects_missing_binding(monkeypatch):
    setup_live(monkeypatch)
    monkeypatch.setattr(live_camera.manager, "latest_bound_frame", lambda: (snapshot(), jpeg(), None))
    response = TestClient(app).post(URL, json=request(), headers={"X-API-Key": "phase40-temp-key"})
    assert response.status_code == 409


def test_http_rejects_unverified_optics(monkeypatch):
    profile = camera_record()
    profile["calibration"]["status"] = "UNCALIBRATED"
    setup_live(monkeypatch, profile=profile)
    response = TestClient(app).post(URL, json=request(), headers={"X-API-Key": "phase40-temp-key"})
    assert response.status_code == 409


def test_http_registry_mutation_changes_only_camera_location_not_pinned_bearing(monkeypatch):
    profile = camera_record(yaw=270)
    setup_live(monkeypatch, profile=profile, frozen=freeze_session_calibration(camera_record(yaw=90), "camA", SESSION))
    response = TestClient(app).post(URL, json=request(), headers={"X-API-Key": "phase40-temp-key"})
    assert response.status_code == 200, response.text
    obs = response.json()["observations"][0]
    assert obs["azimuth_degrees"] == pytest.approx(90)
    assert response.json()["camera_location_bound_to_session_verified"] is False


def test_http_magnetic_request_does_not_draw_guide(monkeypatch):
    setup_live(monkeypatch)
    response = TestClient(app).post(URL, json=request(north_reference="magnetic_north"), headers={"X-API-Key": "phase40-temp-key"})
    assert response.status_code == 200, response.text
    assert len(response.json()["geojson"]["features"]) == 1


def test_http_rejects_mismatched_registry_profile(monkeypatch):
    setup_live(monkeypatch, profile={**camera_record(), "camera_id": "wrong"},
               frozen=freeze_session_calibration(camera_record(), "camA", SESSION))
    response = TestClient(app).post(URL, json=request(), headers={"X-API-Key": "phase40-temp-key"})
    assert response.status_code == 409
