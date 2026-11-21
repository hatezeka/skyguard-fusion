from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

API_KEY = "test-only-secret-key-37450"
HEADERS = {"X-API-Key": API_KEY}

GEO_PAYLOAD = {
    "camera_id": "GEO-SECURITY-001",
    "latitude": 42.6629,
    "longitude": 21.1655,
    "heading_degrees": 90,
    "horizontal_fov_degrees": 80,
    "coverage_radius_m": 500,
    "position_source": "manual",
    "position_accuracy_m": 5,
}


def test_geo_list_rejects_missing_api_key():
    response = client.get("/api/v1/geo/cameras")
    assert response.status_code == 401


def test_geo_list_rejects_wrong_api_key():
    response = client.get(
        "/api/v1/geo/cameras",
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_geo_list_accepts_correct_api_key():
    response = client.get(
        "/api/v1/geo/cameras",
        headers=HEADERS,
    )
    assert response.status_code == 200


def test_geo_write_rejects_missing_api_key():
    response = client.put(
        "/api/v1/geo/cameras/GEO-SECURITY-001",
        json=GEO_PAYLOAD,
    )
    assert response.status_code == 401


def test_geo_write_accepts_correct_api_key():
    response = client.put(
        "/api/v1/geo/cameras/GEO-SECURITY-001",
        json=GEO_PAYLOAD,
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["camera_id"] == "GEO-SECURITY-001"


def test_geo_read_camera_accepts_correct_api_key():
    response = client.get(
        "/api/v1/geo/cameras/GEO-SECURITY-001",
        headers=HEADERS,
    )
    assert response.status_code == 200
