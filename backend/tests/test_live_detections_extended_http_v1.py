from dataclasses import replace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.detection.live_session import SessionSnapshot
from app.main import app


API_KEY = "test-only-secret-key-37450"
URL = "/api/v1/live/detections"


def sample_snapshot():
    return SessionSnapshot(
        state="running",
        frame_index=7,
        detections=[
            {"class": "drone", "confidence": 0.92},
        ],
        tracks=[
            {
                "track_id": 3,
                "class": "drone",
                "state": "confirmed",
            },
        ],
        processing_seconds=0.05,
        error=None,
        camera_id="CAM-001",
        ambiguous_tracks=[
            {
                "class": "unknown",
                "state": "ambiguous",
                "candidate_track_ids": [4, 5],
            },
        ],
        trajectories=[
            {
                "track_id": 3,
                "object_class": "drone",
                "points": [
                    {
                        "frame_index": 7,
                        "x_px": 100.0,
                        "y_px": 80.0,
                    },
                ],
            },
        ],
    )


def test_extended_detections_http_response():
    snapshot = sample_snapshot()

    with patch(
        "app.api.live_camera.manager.status",
        return_value=snapshot,
    ):
        with TestClient(
            app,
            headers={"X-API-Key": API_KEY},
        ) as client:
            response = client.get(URL)

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["state"] == "running"
    assert body["frame_index"] == 7
    assert body["detections"] == snapshot.detections
    assert body["tracks"] == snapshot.tracks
    assert body["ambiguous_tracks"] == snapshot.ambiguous_tracks
    assert body["trajectories"] == snapshot.trajectories
    assert body["processing_seconds"] == 0.05
    assert body["error"] is None


def test_extended_detections_empty_lists():
    snapshot = replace(
        sample_snapshot(),
        ambiguous_tracks=[],
        trajectories=[],
    )

    with patch(
        "app.api.live_camera.manager.status",
        return_value=snapshot,
    ):
        with TestClient(
            app,
            headers={"X-API-Key": API_KEY},
        ) as client:
            response = client.get(URL)

    assert response.status_code == 200, response.text
    assert response.json()["ambiguous_tracks"] == []
    assert response.json()["trajectories"] == []


def test_extended_detections_requires_api_key():
    with patch(
        "app.api.live_camera.manager.status"
    ) as manager_status:
        with TestClient(app) as client:
            response = client.get(URL)

    assert response.status_code in (401, 403)
    manager_status.assert_not_called()
