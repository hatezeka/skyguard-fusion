"""41.2: HTTP route, inherited auth, idempotency and absence of target GPS."""
from __future__ import annotations

from dataclasses import dataclass
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import live_camera

URL = '/api/v1/live/alerts/evaluate/latest'
KEY = 'phase-41-2-test-key-only'
SID_A = 'a' * 32
SID_B = 'b' * 32


@dataclass(frozen=True)
class Snapshot:
    state: str
    session_id: str | None
    camera_id: str | None
    frame_index: int | None
    tracks: list
    ambiguous_tracks: list


def sample(frame=1, sid=SID_A, state='running', label='drone', camera='cam_1'):
    return Snapshot(
        state=state, session_id=sid, camera_id=camera, frame_index=frame,
        tracks=[{'track_id': 9, 'class': label, 'confidence': 0.95,
                 'state': 'confirmed', 'frame_index': frame}],
        ambiguous_tracks=[],
    )


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch):
    monkeypatch.setenv('SKYGUARD_API_KEY', KEY)
    live_camera._live_advisory_bridge.reset()
    yield
    live_camera._live_advisory_bridge.reset()


def mock_frame(monkeypatch, frame=1, **kwargs):
    state = {'value': sample(frame, **kwargs)}
    monkeypatch.setattr(live_camera.manager, 'status', lambda: state['value'])
    return state


def client():
    return TestClient(app)


def auth():
    return {'X-API-Key': KEY}


def test_effective_openapi_path_present():
    assert 'post' in app.openapi().get('paths', {}).get(URL, {})


def test_request_requires_auth(monkeypatch):
    mock_frame(monkeypatch)
    with client() as http:
        assert http.post(URL).status_code in (401, 403)


def test_inactive_returns_409(monkeypatch):
    mock_frame(monkeypatch, state='stopped')
    with client() as http:
        response = http.post(URL, headers=auth())
    assert response.status_code == 409
    assert 'tracks' not in response.json()


def test_server_selects_frames_not_client(monkeypatch):
    state = mock_frame(monkeypatch)
    with client() as http:
        first = http.post(URL, headers=auth())
        state['value'] = sample(2)
        second = http.post(URL, headers=auth())
    assert first.status_code == second.status_code == 200
    assert first.json()['frame_index'] == 1
    assert second.json()['frame_index'] == 2
    assert first.json()['alerts'] == second.json()['alerts'] == []


def test_http_emits_one_model_advisory_and_replay_stable(monkeypatch):
    state = mock_frame(monkeypatch)
    with client() as http:
        bodies = []
        for frame in (1, 2, 3):
            state['value'] = sample(frame)
            r = http.post(URL, headers=auth())
            assert r.status_code == 200, r.text
            bodies.append(r.json())
        repeat = http.post(URL, headers=auth())
        state['value'] = sample(4)
        later = http.post(URL, headers=auth())
    assert not bodies[0]['alerts'] and not bodies[1]['alerts']
    assert len(bodies[2]['alerts']) == 1
    assert bodies[2] == repeat.json()
    assert later.json()['alerts'] == []
    assert bodies[2]['alerts'][0]['event_id']
    assert bodies[2]['automated_response_allowed'] is False
    assert bodies[2]['target_gps_available'] is False


def test_http_birds_never_alert(monkeypatch):
    state = mock_frame(monkeypatch, label='bird')
    with client() as http:
        for frame in (1, 2, 3, 4):
            state['value'] = sample(frame, label='bird')
            r = http.post(URL, headers=auth())
            assert r.status_code == 200
            assert r.json()['alerts'] == []


def test_restarts_reset_alert_observations(monkeypatch):
    state = mock_frame(monkeypatch)
    with client() as http:
        for frame in (1, 2, 3):
            state['value'] = sample(frame)
            before = http.post(URL, headers=auth()).json()
        state['value'] = sample(1, sid=SID_B)
        first_after = http.post(URL, headers=auth()).json()
    assert len(before['alerts']) == 1
    assert first_after['alerts'] == []
    assert first_after['session_id'] == SID_B


def test_untrusted_json_does_not_supply_alert_evidence(monkeypatch):
    state = mock_frame(monkeypatch, label='bird')
    spoof = {'session_id': SID_A, 'frame_index': 999999,
             'tracks':[{'class':'drone','confidence':1.0,'track_id':9,'state':'confirmed','frame_index':999999}]}
    with client() as http:
        # Endpoint intentionally has no request body; body is ignored rather than
        # trusted. Response MUST only reflect backend manager.status().
        r = http.post(URL, json=spoof, headers=auth())
    assert r.status_code == 200
    assert r.json()['frame_index'] == 1
    assert not r.json()['alerts']
