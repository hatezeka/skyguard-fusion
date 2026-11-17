from fastapi.testclient import TestClient
from app.main import app
from app.schemas import CameraProfile, Detection, PositionEstimate
from pydantic import ValidationError
import pytest

client = TestClient(app)

def test_health():
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_camera_list_starts_empty():
    assert client.get('/api/v1/cameras').json() == []

def test_standard_profile_fallback():
    p = CameraProfile(camera_id='cam-1', width=640, height=480, fps=25)
    assert p.profile_id == 'standard'
    assert p.profile_source == 'fallback'

def test_invalid_detection_box():
    with pytest.raises(ValidationError):
        Detection(detection_id='d1', camera_id='c1', frame_index=1, timestamp_ms=1,
                  class_name='drone', confidence=.9, bbox_xyxy=(10,10,5,20))

def test_bearing_only_does_not_claim_distance():
    with pytest.raises(ValidationError):
        PositionEstimate(method='bearing_only', bearing_deg=40, distance_m=150)
