import base64

from fastapi.testclient import TestClient

from app.main import app
from app.api import live_camera
from app.detection.live_session import SessionSnapshot


def snapshot(state="running", frame_index=42):
    return SessionSnapshot(
        state=state,
        frame_index=frame_index,
        detections=[],
        tracks=[{
            "track_id": 3,
            "class": "drone",
            "state": "confirmed",
            "frame_index": frame_index,
        }],
        processing_seconds=0.01,
        error=None,
        trajectories=[{
            "track_id": 3,
            "object_class": "drone",
            "points": [],
        }],
    )


def test_synchronized_frame(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    expected = snapshot()

    monkeypatch.setattr(
        live_camera.manager,
        "latest_frame",
        lambda: (expected, b"jpeg-frame-42"),
    )

    client = TestClient(app)
    response = client.get(
        "/api/v1/live/frame",
        headers={"X-API-Key": "test-secret"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["frame_index"] == 42
    assert data["image_mime"] == "image/jpeg"
    assert base64.b64decode(data["image_base64"]) == b"jpeg-frame-42"
    assert data["tracks"][0]["frame_index"] == 42
    assert data["trajectories"][0]["track_id"] == 3


def test_inactive_frame(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    monkeypatch.setattr(
        live_camera.manager,
        "latest_frame",
        lambda: (snapshot("stopped"), b"jpeg"),
    )

    response = TestClient(app).get(
        "/api/v1/live/frame",
        headers={"X-API-Key": "test-secret"},
    )

    assert response.status_code == 409


def test_missing_jpeg(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")
    monkeypatch.setattr(
        live_camera.manager,
        "latest_frame",
        lambda: (snapshot(), None),
    )

    response = TestClient(app).get(
        "/api/v1/live/frame",
        headers={"X-API-Key": "test-secret"},
    )

    assert response.status_code == 409


def test_missing_api_key(monkeypatch):
    monkeypatch.setenv("SKYGUARD_API_KEY", "test-secret")

    response = TestClient(app).get("/api/v1/live/frame")

    assert response.status_code == 401
