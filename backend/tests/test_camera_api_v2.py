import os
from fastapi.testclient import TestClient
from app.main import app
from app.api.camera_management import registry

def test_camera_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv('SKYGUARD_CAMERA_REGISTRY', str(tmp_path / 'profiles.json'))
    registry.cache_clear()
    with TestClient(app, headers={'X-API-Key': 'test-only-secret-key-37450'}) as client:
        assert client.get('/api/v1/health').status_code == 200
        base = '/api/v1/camera-profiles'
        r = client.post(base, json={'camera_id':'CAM-001','source_type':'usb','stream':{'width_px':1280,'height_px':720,'fps':30}})
        assert r.status_code == 201, r.text
        assert client.post(base, json={'camera_id':'CAM-001','source_type':'usb','stream':{'width_px':1280,'height_px':720,'fps':30}}).status_code == 409
        assert len(client.get(base).json()) == 1
        assert client.get(base+'/missing').status_code == 404
        assert client.get(base+'/CAM-001/readiness').json()['ray_geometry_ready'] is False
        loc = {'latitude':42.6,'longitude':20.3,'altitude_m':550,'source':'manual','horizontal_accuracy_m':10}
        assert client.patch(base+'/CAM-001/location', json=loc).status_code == 200
        intr = {'fx':800,'fy':800,'cx':640,'cy':360,'distortion':[0,0,0,0,0]}
        assert client.patch(base+'/CAM-001/calibration', json=intr).status_code == 200
        assert client.patch(base+'/CAM-001/orientation', json={'yaw_deg':0,'pitch_deg':0,'roll_deg':0,'source':'manual'}).status_code == 200
        ready = client.get(base+'/CAM-001/readiness').json()
        assert ready['ray_geometry_ready'] is True
        assert ready['camera_geolocation_ready'] is True
        assert ready['drone_depth_observable'] is False
        assert client.put(base+'/CAM-001/stream',json={'width_px':1920,'height_px':1080,'fps':30}).status_code == 200
        assert client.get(base+'/CAM-001').json()['calibration']['status'] == 'INVALIDATED'
    registry.cache_clear()
