import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.camera_management import registry


CAMERA_PAYLOAD = {
    "camera_id": "AUTH-TDD-001",
    "source_type": "usb",
    "source_location": 0,
    "stream": {
        "width_px": 640,
        "height_px": 480,
        "fps": 30,
    },
}


@pytest.fixture
def isolated_registry():
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(
            os.environ,
            {
                "SKYGUARD_CAMERA_REGISTRY":
                    str(Path(directory) / "registry.json"),
                "SKYGUARD_API_KEY":
                    "test-only-secret-key-37447",
            },
        ):
            registry.cache_clear()

            try:
                with TestClient(app) as client:
                    yield client
            finally:
                registry.cache_clear()


def test_registration_without_key_is_rejected(isolated_registry):
    response = isolated_registry.post(
        "/api/v1/camera-profiles",
        json=CAMERA_PAYLOAD,
    )

    assert response.status_code == 401


def test_registration_with_wrong_key_is_rejected(isolated_registry):
    response = isolated_registry.post(
        "/api/v1/camera-profiles",
        headers={"X-API-Key": "incorrect-key"},
        json=CAMERA_PAYLOAD,
    )

    assert response.status_code == 401


def test_registration_with_correct_key_is_allowed(isolated_registry):
    response = isolated_registry.post(
        "/api/v1/camera-profiles",
        headers={"X-API-Key": "test-only-secret-key-37447"},
        json=CAMERA_PAYLOAD,
    )

    assert response.status_code == 201


def test_live_status_without_key_is_rejected(isolated_registry):
    response = isolated_registry.get(
        "/api/v1/live/status",
    )

    assert response.status_code == 401


def test_health_remains_public(isolated_registry):
    response = isolated_registry.get(
        "/api/v1/health",
    )

    assert response.status_code == 200


def test_missing_server_key_fails_closed(tmp_path):
    registry_path = str(tmp_path / "registry.json")

    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "SKYGUARD_API_KEY"
    }

    environment["SKYGUARD_CAMERA_REGISTRY"] = registry_path

    with patch.dict(os.environ, environment, clear=True):
        registry.cache_clear()

        try:
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/camera-profiles",
                    json=CAMERA_PAYLOAD,
                )

                assert response.status_code in (401, 503)
        finally:
            registry.cache_clear()
