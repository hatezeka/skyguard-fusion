"""Camera geospatial API for SkyGuard FUSION."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from threading import RLock
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.localization.camera_profile import CameraGeoProfile, utc_now

router = APIRouter(prefix="/api/v1/geo", tags=["geolocation"])

_profiles: dict[str, CameraGeoProfile] = {}
_lock = RLock()


class CameraGeoInput(BaseModel):
    camera_id: str = Field(min_length=1, max_length=128)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    heading_degrees: float = Field(ge=0, lt=360, allow_inf_nan=False)
    horizontal_fov_degrees: float = Field(gt=0, le=180, allow_inf_nan=False)
    coverage_radius_m: float = Field(gt=0, le=100000, allow_inf_nan=False)
    position_source: Literal["manual", "gnss"] = "manual"
    position_accuracy_m: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )


def serialize_profile(profile: CameraGeoProfile) -> dict:
    data = asdict(profile)
    data["coverage_polygon"] = [
        {"latitude": lat, "longitude": lon}
        for lat, lon in profile.coverage_polygon()
    ]
    return data


@router.put("/cameras/{camera_id}")
def upsert_camera_profile(
    camera_id: str,
    payload: CameraGeoInput,
):
    if camera_id != payload.camera_id:
        raise HTTPException(
            status_code=422,
            detail="Camera ID in URL and body must match",
        )

    try:
        profile = CameraGeoProfile(
            **payload.model_dump(),
            updated_at=utc_now(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _lock:
        _profiles[camera_id] = profile

    return serialize_profile(profile)


@router.get("/cameras")
def list_camera_profiles():
    with _lock:
        profiles = list(_profiles.values())

    return [serialize_profile(p) for p in profiles]


@router.get("/cameras/{camera_id}")
def get_camera_profile(camera_id: str):
    with _lock:
        profile = _profiles.get(camera_id)

    if profile is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    return serialize_profile(profile)
