"""40.2 atomic latest scene, authorization, and explicit no-GPS properties."""
from __future__ import annotations

from copy import deepcopy
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import live_camera
from app.detection.live_session import SessionSnapshot
from app.localization.session_calibration import freeze_session_calibration

SID = '0123456789abcdef0123456789abcdef'
URL = '/api/v1/live/map/scene/latest'
KEY = 'phase-40-2-ephemeral'


def record():
    return {
        'camera_id':'cam40','source_type':'usb','stream':{'width_px':160,'height_px':120},
        'calibration':{'status':'CALIBRATED','resolution':[160,120],
            'verified_at':'2026-11-20T00:00:00Z',
            'intrinsics':{'fx':110.0,'fy':110.0,'cx':80.0,'cy':60.0,'distortion':[0.0]*5}},
        'orientation':{'yaw_deg':90.0,'pitch_deg':0.0,'roll_deg':0.0,'source':'manual'},
        'geolocation':{'source':'manual','latitude':42.66,'longitude':21.16,
           'horizontal_accuracy_m':None,'captured_at':None},
    }


def snapshot(camera='cam40', session=SID, frame=123, state='running'):
    return SessionSnapshot(state, frame, [], [], .01, None, camera_id=camera, session_id=session)


def jpeg():
    ok, result=cv2.imencode('.jpg',np.zeros((120,160,3),dtype=np.uint8))
    assert ok
    return result.tobytes()


def setup(monkeypatch, *, snap=None, binding=True, profile=None):
    monkeypatch.setenv('SKYGUARD_API_KEY',KEY)
    rec=record() if profile is None else profile
    freeze=freeze_session_calibration(rec, 'cam40', SID) if binding else None
    monkeypatch.setattr(live_camera.manager,'latest_bound_frame',lambda: (
        snapshot() if snap is None else snap, jpeg(), freeze,
    ))
    monkeypatch.setattr(live_camera,'registry',lambda: type('Registry',(),{
        'get':lambda self,camera_id:deepcopy(rec),
    })())


def test_openapi_contains_effective_latest_route():
    assert 'get' in app.openapi()['paths'][URL]


def test_auth_required(monkeypatch):
    setup(monkeypatch)
    assert TestClient(app).get(URL,params={'north_reference':'magnetic_north'}).status_code in (401,403)


def test_required_north_reference(monkeypatch):
    setup(monkeypatch)
    assert TestClient(app).get(URL,headers={'X-API-Key':KEY}).status_code == 422


def test_atomic_scene_binds_server_selected_frame(monkeypatch):
    setup(monkeypatch)
    result=TestClient(app).get(URL,params={'north_reference':'true_north'},headers={'X-API-Key':KEY})
    assert result.status_code == 200,result.text
    body=result.json()
    assert body['session_id']==SID and body['frame_index']==123
    assert body['target_gps_available'] is False
    assert body['target_range_available'] is False
    assert body['calibration_bound_to_frame_verified'] is True
    assert body['camera_location_bound_to_session_verified'] is False
    assert body['physical_calibration_verified'] is False


def test_magnetic_north_suppresses_guides(monkeypatch):
    setup(monkeypatch)
    result=TestClient(app).get(URL,params={'north_reference':'magnetic_north'},headers={'X-API-Key':KEY})
    assert result.status_code==200,result.text
    assert all(f['properties']['kind']!='bearing_guide' for f in result.json()['geojson']['features'])


@pytest.mark.parametrize('snap',[
    snapshot(state='stopped'),snapshot(camera=None),
    snapshot(session=None),snapshot(frame=None),
])
def test_bad_active_frame_fails_closed(monkeypatch,snap):
    setup(monkeypatch,snap=snap)
    result=TestClient(app).get(URL,params={'north_reference':'true_north'},headers={'X-API-Key':KEY})
    assert result.status_code == 409


def test_missing_calibration_fails_closed(monkeypatch):
    setup(monkeypatch,binding=False)
    result=TestClient(app).get(URL,params={'north_reference':'true_north'},headers={'X-API-Key':KEY})
    assert result.status_code == 409


def test_mismatched_identity_fails_closed(monkeypatch):
    setup(monkeypatch,snap=snapshot(camera='wrong'))
    result=TestClient(app).get(URL,params={'north_reference':'true_north'},headers={'X-API-Key':KEY})
    assert result.status_code == 409
