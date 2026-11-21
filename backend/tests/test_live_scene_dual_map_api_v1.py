"""42.6D effective protected live-map route remains the ONE existing route."""
from __future__ import annotations
import ast
from pathlib import Path


def test_single_existing_live_scene_route_still_declared():
    from app.main import app
    paths = app.openapi().get('paths', {})
    endpoint = '/api/v1/live/map/scene'
    assert 'post' in paths.get(endpoint, {}), 'Existing protected map scene endpoint missing'
    assert '/api/v1/live/map/dual' not in paths, 'Duplicate map API must not be created'


def test_live_scene_api_has_internal_dual_map_wiring():
    import app.api.live_camera as live_camera
    path=Path(live_camera.__file__)
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    matches=[]
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if (isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute)
                and d.func.attr=='post' and d.args
                and isinstance(d.args[0],ast.Constant)
                and d.args[0].value=='/map/scene'):
                matches.append(node)
    assert len(matches)==1, 'Exactly one existing route must be wired'
    calls=[n for n in ast.walk(matches[0]) if isinstance(n,ast.Call)]
    assert sum(isinstance(n.func,ast.Name) and n.func.id=='attach_dual_map_state' for n in calls)==1
    assert sum(isinstance(n.func,ast.Name) and n.func.id=='build_live_map_scene' for n in calls)==1


def test_actual_protected_http_scene_includes_safe_dual_map(monkeypatch):
    """One real TestClient roundtrip through the EXISTING auth-protected route."""
    from fastapi.testclient import TestClient
    import cv2
    import numpy as np
    from app.main import app
    from app.api import live_camera
    from app.detection.live_session import SessionSnapshot
    from app.localization.session_calibration import freeze_session_calibration
    import copy

    sid='0123456789abcdef0123456789abcdef'
    camera={
        'camera_id':'camA',
        'source_location':'rtsp://private.example/camera',
        'stream':{'width_px':200,'height_px':100},
        'calibration':{'status':'CALIBRATED','resolution':[200,100],
            'verified_at':'2026-11-20T14:00:00Z',
            'intrinsics':{'fx':100.0,'fy':100.0,'cx':100.0,'cy':50.0,
                         'distortion':[0.0]*5}},
        'orientation':{'yaw_deg':90.0,'pitch_deg':0.0,'roll_deg':0.0,'source':'manual'},
        'geolocation':{'latitude':42.66,'longitude':21.16,'source':'manual',
            'horizontal_accuracy_m':None,'captured_at':None},
    }
    frozen=freeze_session_calibration(camera,'camA',sid)
    snap=SessionSnapshot('running',7,
        [{'class':'drone','confidence':.9,'bbox':{'x':90,'y':40,'width':20,'height':20}}],
        [],.01,None,camera_id='camA',session_id=sid)
    ok, encoded=cv2.imencode('.jpg',np.zeros((100,200,3),dtype=np.uint8))
    assert ok
    monkeypatch.setenv('SKYGUARD_API_KEY','phase42-temp-key')
    monkeypatch.setattr(live_camera.manager,'latest_bound_frame',
                        lambda:(snap,encoded.tobytes(),frozen))
    monkeypatch.setattr(live_camera,'registry',
        lambda:type('MockRegistry',(),{'get':lambda self,camera_id:copy.deepcopy(camera)})())
    request={'camera_id':'camA','expected_frame_index':7,
             'expected_session_id':sid,'north_reference':'true_north'}
    response=TestClient(app).post('/api/v1/live/map/scene',json=request,
                                  headers={'X-API-Key':'phase42-temp-key'})
    assert response.status_code==200,response.text
    body=response.json()
    assert body['scene_type']=='live_camera_bearing_map'
    assert body['dual_map_v1']['session_id']==sid
    assert body['dual_map_v1']['frame_index']==7
    assert len(body['dual_map_v1']['tactical_map']['objects'])==1
    assert body['dual_map_v1']['tactical_map']['objects'][0]['track_id'] is None
    assert body['dual_map_v1']['real_map']['verified_estimate_geojson']['features']==[]
    assert body['target_gps_available'] is False
    assert body['target_range_available'] is False