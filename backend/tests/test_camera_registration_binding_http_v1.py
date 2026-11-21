import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.camera_management import registry


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SKYGUARD_CAMERA_REGISTRY",
        str(tmp_path / "camera_profiles.json"),
    )
    registry.cache_clear()

    try:
        with TestClient(app, headers={'X-API-Key': 'test-only-secret-key-37450'}) as test_client:
            yield test_client
    finally:
        registry.cache_clear()


def payload(camera_id, source_type, source_location=None):
    return {
        "camera_id": camera_id,
        "source_type": source_type,
        "source_location": source_location,
        "stream": {
            "width_px": 640,
            "height_px": 480,
            "fps": 30,
        },
    }


def test_register_usb_binding_http(client):
    response = client.post(
        "/api/v1/camera-profiles",
        json=payload("CAM-USB-HTTP", "usb", 0),
    )

    assert response.status_code == 201, response.text
    assert response.json()["source_location"] == 0

    fetched = client.get(
        "/api/v1/camera-profiles/CAM-USB-HTTP"
    )

    assert fetched.status_code == 200
    assert fetched.json()["source_location"] == 0


def test_register_rtsp_binding_http(client):
    response = client.post(
        "/api/v1/camera-profiles",
        json=payload(
            "CAM-RTSP-HTTP",
            "rtsp",
            "rtsp://camera.local/live",
        ),
    )

    assert response.status_code == 201, response.text
    assert (
        response.json()["source_location"]
        == "rtsp://camera.local/live"
    )


@pytest.mark.parametrize(
    "source_type,source_location",
    [
        ("usb", -1),
        ("usb", "0"),
        ("rtsp", "http://camera.local/live"),
        ("rtsp", "rtsp://"),
        ("rtsp", "rtsp://admin:secret@camera.local/live"),
    ],
)
def test_invalid_binding_rejected_http(
    client,
    source_type,
    source_location,
):
    response = client.post(
        "/api/v1/camera-profiles",
        json=payload(
            "CAM-INVALID",
            source_type,
            source_location,
        ),
    )

    assert response.status_code == 422
    assert "secret" not in response.text


def test_legacy_registration_http(client):
    data = payload("CAM-LEGACY-HTTP", "usb")
    data.pop("source_location")

    response = client.post(
        "/api/v1/camera-profiles",
        json=data,
    )

    assert response.status_code == 201
    assert response.json().get("source_location") is None
