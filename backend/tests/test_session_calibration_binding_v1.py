"""Phase 39 closure: immutable, session-bound optical profile and atomic frame."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import live_camera, camera_management
from app.detection.live_session import LiveSessionManager, SessionSnapshot
from app.localization.session_calibration import freeze_session_calibration
from app.main import app

SESSION = "0123456789abcdef0123456789abcdef"
OTHER = "fedcba9876543210fedcba9876543210"
URL = "/api/v1/live/frame/bearings/registered/pinned"


def profile(*, yaw=90.0, calibration_status="CALIBRATED"):
    return {
        "camera_id": "camA",
        "source_location": "rtsp://username:password@nowhere:554/secret?token=abc",
        "geolocation": {"latitude": 42.0, "longitude": 21.0},
        "password": "do-not-store",
        "stream": {"width_px": 200, "height_px": 100, "fps": 25.0},
        "calibration": {
            "status": calibration_status,
            "resolution": [200, 100] if calibration_status == "CALIBRATED" else None,
            "verified_at": "2026-11-20T14:00:00Z" if calibration_status == "CALIBRATED" else None,
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 100.0,
                           "cy": 50.0, "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]}
            if calibration_status == "CALIBRATED" else None,
        },
        "orientation": {"yaw_deg": yaw, "pitch_deg": 0.0,
                        "roll_deg": 0.0, "source": "manual"},
    }


def jpeg(w=200, h=100):
    ok, data = cv2.imencode('.jpg', np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    return data.tobytes()


def snap(*, session=SESSION, camera='camA', index=42, state='running', detections=None):
    return SessionSnapshot(
        state, index,
        detections if detections is not None else [
            {"class": "drone", "confidence": 0.9,
             "bbox": {"x": 90, "y": 40, "width": 20, "height": 20}}
        ],
        [], 0.01, None, camera_id=camera, session_id=session,
    )


def req(**change):
    data = {"camera_id": "camA", "expected_frame_index": 42,
            "expected_session_id": SESSION, "north_reference": "true_north"}
    data.update(change)
    return data


def freeze(source=None, session=SESSION):
    return freeze_session_calibration(source or profile(), "camA", session)


def test_freeze_whitelisted_fields_only():
    record = profile()
    record['stream']['extra_secret'] = 'private-stream-secret'
    record['calibration']['private_key'] = 'private-cal-secret'
    record['orientation']['password'] = 'private-pose-secret'
    bound = freeze(record)
    assert 'private-stream-secret' not in bound.canonical_json
    assert 'private-cal-secret' not in bound.canonical_json
    assert 'private-pose-secret' not in bound.canonical_json
    assert set(bound.record()) == {"camera_id", "stream", "calibration", "orientation"}
    assert 'password' not in bound.canonical_json
    assert 'rtsp' not in bound.canonical_json
    assert 'geolocation' not in bound.canonical_json
    assert 'token' not in bound.canonical_json


def test_freeze_deep_content_isolation():
    record = profile()
    bound = freeze(record)
    record['calibration']['intrinsics']['fx'] = 999
    record['orientation']['yaw_deg'] = 270
    assert bound.record()['calibration']['intrinsics']['fx'] == 100
    assert bound.record()['orientation']['yaw_deg'] == 90


def test_freeze_record_accessor_cannot_mutate_source():
    bound = freeze()
    mutable = bound.record()
    mutable['calibration']['intrinsics']['distortion'][0] = 666
    assert bound.record()['calibration']['intrinsics']['distortion'][0] == 0


def test_fingerprint_stable_for_equivalent_registry_record():
    a = profile()
    b = deepcopy(a)
    b['stream'] = dict(reversed(list(b['stream'].items())))
    assert freeze(a).sha256 == freeze(b).sha256
    assert len(freeze(a).sha256) == 64


def test_fingerprint_changes_with_orientation():
    assert freeze(profile(yaw=90)).sha256 != freeze(profile(yaw=100)).sha256


def test_fingerprint_validated_on_read():
    value = freeze()
    altered = replace(value, sha256="0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        altered.record()


def test_freeze_refuses_wrong_camera():
    with pytest.raises(ValueError, match="identity"):
        freeze_session_calibration(profile(), "camWrong", SESSION)


def test_freeze_refuses_nan_json():
    record = profile()
    record['orientation']['yaw_deg'] = float('nan')
    with pytest.raises(ValueError, match="cannot be frozen"):
        freeze_session_calibration(record, "camA", SESSION)


def test_freeze_refuses_nonserializable_data():
    record = profile()
    record['orientation']['yaw_deg'] = object()
    with pytest.raises(ValueError, match="cannot be frozen"):
        freeze_session_calibration(record, "camA", SESSION)


def test_missing_record_allows_detection_only():
    assert freeze_session_calibration(None, None, SESSION) is None


def test_uncalibrated_record_frozen_even_if_later_modified():
    record = profile(calibration_status="UNCALIBRATED")
    bound = freeze(record)
    record['calibration'] = profile()['calibration']
    assert bound.record()['calibration']['status'] == "UNCALIBRATED"


def test_manager_restart_rebinds_new_session(monkeypatch):
    class NoopThread:
        def __init__(self, *args, **kwargs): pass
        def is_alive(self): return False
        def start(self): pass
    monkeypatch.setattr('app.detection.live_session.threading.Thread', NoopThread)
    mgr = LiveSessionManager()
    mgr.start(None, None, camera_id="camA", calibration_record=profile(yaw=90))
    first, _, bound = mgr.latest_bound_frame()
    assert first.state == 'starting'
    assert bound.session_id == first.session_id
    assert bound.record()['orientation']['yaw_deg'] == 90
    mgr.start(None, None, camera_id="camA", calibration_record=profile(yaw=100))
    second, _, rebound = mgr.latest_bound_frame()
    assert second.session_id != first.session_id
    assert rebound.session_id == second.session_id
    assert rebound.record()['orientation']['yaw_deg'] == 100
    assert bound.record()['orientation']['yaw_deg'] == 90


def test_manager_frame_snapshot_and_binding_are_one_locked_tuple():
    mgr = LiveSessionManager()
    frozen = freeze()
    s = snap()
    with mgr._lock:
        mgr._snapshot = s
        mgr._jpeg = b'frame'
        mgr._bound_calibration = frozen
    result = mgr.latest_bound_frame()
    assert result == (s, b'frame', frozen)


def test_manager_start_rejects_wrong_camera_without_state_change(monkeypatch):
    class NoopThread:
        def __init__(self, *args, **kwargs): pass
        def is_alive(self): return False
        def start(self): pass
    monkeypatch.setattr('app.detection.live_session.threading.Thread', NoopThread)
    mgr = LiveSessionManager()
    with pytest.raises(ValueError, match="identity"):
        mgr.start(None, None, camera_id="other", calibration_record=profile())
    assert mgr.status().state == "idle"


def test_pinned_http_success_preserves_physical_unverified(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "local-test-key")
    frozen = freeze()
    monkeypatch.setattr(live_camera.manager, "latest_bound_frame", lambda: (snap(), jpeg(), frozen))
    monkeypatch.setattr(camera_management, 'registry', lambda: (_ for _ in ()).throw(AssertionError('must never reload registry')))
    result = TestClient(app).post(URL, json=req(), headers={"X-API-Key": "local-test-key"})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body['session_id'] == SESSION
    assert body['calibration_snapshot_sha256'] == frozen.sha256
    assert body['profile_source'] == 'camera_registry_session_snapshot'
    assert body['profile_bound_to_session_verified'] is True
    assert body['profile_bound_to_frame_verified'] is True
    assert body['physical_calibration_verified'] is False
    assert body['physical_distance_available'] is False
    assert body['target_gps_available'] is False
    assert body['directions'][0]['direction']['azimuth_degrees'] == pytest.approx(90)


@pytest.mark.parametrize('change', [
    {'session': OTHER}, {'camera': 'other'}, {'index': 41}, {'state': 'stopped'},
])
def test_pinned_http_rejects_stale_frame_or_session(monkeypatch, change):
    monkeypatch.setenv('SKYGUARD_API_KEY', 'local-test-key')
    frozen = freeze()
    monkeypatch.setattr(live_camera.manager, 'latest_bound_frame', lambda: (
        snap(**change), jpeg(), frozen
    ))
    response = TestClient(app).post(URL, json=req(), headers={'X-API-Key': 'local-test-key'})
    assert response.status_code == 409


def test_pinned_http_rejects_wrong_binding_session(monkeypatch):
    monkeypatch.setenv('SKYGUARD_API_KEY', 'local-test-key')
    frozen = freeze(session=OTHER)
    monkeypatch.setattr(live_camera.manager, 'latest_bound_frame', lambda: (snap(), jpeg(), frozen))
    response = TestClient(app).post(URL, json=req(), headers={'X-API-Key': 'local-test-key'})
    assert response.status_code == 409


def test_pinned_http_rejects_absent_calibration_snapshot(monkeypatch):
    monkeypatch.setenv('SKYGUARD_API_KEY', 'local-test-key')
    monkeypatch.setattr(live_camera.manager, 'latest_bound_frame', lambda: (snap(), jpeg(), None))
    response = TestClient(app).post(URL, json=req(), headers={'X-API-Key': 'local-test-key'})
    assert response.status_code == 409


def test_pinned_http_never_retrospectively_calibrates(monkeypatch):
    monkeypatch.setenv('SKYGUARD_API_KEY', 'local-test-key')
    frozen = freeze(profile(calibration_status='UNCALIBRATED'))
    monkeypatch.setattr(live_camera.manager, 'latest_bound_frame', lambda: (snap(), jpeg(), frozen))
    response = TestClient(app).post(URL, json=req(), headers={'X-API-Key': 'local-test-key'})
    assert response.status_code == 409


def test_pinned_http_rejects_bad_jpeg_dimensions(monkeypatch):
    monkeypatch.setenv('SKYGUARD_API_KEY', 'local-test-key')
    monkeypatch.setattr(live_camera.manager, 'latest_bound_frame', lambda: (snap(), jpeg(250,100), freeze()))
    response = TestClient(app).post(URL, json=req(), headers={'X-API-Key': 'local-test-key'})
    assert response.status_code == 409


def test_pinned_http_rejects_missing_session_id(monkeypatch):
    monkeypatch.setenv('SKYGUARD_API_KEY', 'local-test-key')
    bad = req()
    del bad['expected_session_id']
    response = TestClient(app).post(URL, json=bad, headers={'X-API-Key': 'local-test-key'})
    assert response.status_code == 422


def test_pinned_http_auth_rejects_no_key(monkeypatch):
    monkeypatch.setenv('SKYGUARD_API_KEY', 'local-test-key')
    response = TestClient(app).post(URL, json=req())
    assert response.status_code in (401,403)


def test_pinned_is_in_openapi():
    assert 'post' in app.openapi()['paths'][URL]
