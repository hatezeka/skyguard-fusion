import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.live_camera import StartRequest, resolve_live_source


def test_legacy_usb_request():
    source = resolve_live_source(StartRequest(camera_index=0))
    assert source.source_type == "usb"
    assert source.location == 0


def test_explicit_usb_index():
    source = resolve_live_source(
        StartRequest(source_type="usb", source_location=2)
    )
    assert source.location == 2


@pytest.mark.parametrize("location", [-1, 5, "0"])
def test_invalid_usb_location(location):
    with pytest.raises(HTTPException) as exc:
        resolve_live_source(
            StartRequest(source_type="usb", source_location=location)
        )
    assert exc.value.status_code == 422


def test_boolean_usb_location_rejected():
    with pytest.raises(ValidationError):
        StartRequest(source_type="usb", source_location=True)


def test_rtsp_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SKYGUARD_ENABLE_RTSP_LIVE", raising=False)

    with pytest.raises(HTTPException) as exc:
        resolve_live_source(
            StartRequest(
                source_type="rtsp",
                source_location="rtsp://camera.example/live",
            )
        )

    assert exc.value.status_code == 403


def test_rtsp_enabled(monkeypatch):
    monkeypatch.setenv("SKYGUARD_ENABLE_RTSP_LIVE", "1")

    source = resolve_live_source(
        StartRequest(
            source_type="rtsp",
            source_location="rtsp://camera.example/live",
        )
    )

    assert source.source_type == "rtsp"


def test_invalid_rtsp_url(monkeypatch):
    monkeypatch.setenv("SKYGUARD_ENABLE_RTSP_LIVE", "1")

    with pytest.raises(HTTPException) as exc:
        resolve_live_source(
            StartRequest(
                source_type="rtsp",
                source_location="http://camera.example/live",
            )
        )

    assert exc.value.status_code == 422


def test_unsupported_source():
    with pytest.raises(HTTPException) as exc:
        resolve_live_source(StartRequest(source_type="unknown"))

    assert exc.value.status_code == 422
