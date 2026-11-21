"""Camera profile API; no physical camera or GPS discovery is implied."""
from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.camera_management.core import CameraRegistry, CameraError

router = APIRouter(prefix='/api/v1/camera-profiles', tags=['Camera management'])

@lru_cache(maxsize=1)
def registry():
    path = Path(os.environ.get('SKYGUARD_CAMERA_REGISTRY', 'data/camera_profiles.json'))
    return CameraRegistry(path)

class Stream(BaseModel):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    fps: float = Field(gt=0)

class Registration(BaseModel):
    camera_id: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    source_type: str
    stream: Stream
    source_location: int | str | None = None

class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    source: str
    altitude_m: float | None = None
    horizontal_accuracy_m: float | None = Field(default=None, ge=0)

class Intrinsics(BaseModel):
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    distortion: list[float] = Field(min_length=5, max_length=5)

class Orientation(BaseModel):
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    source: str

def invoke(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except CameraError as exc:
        message = str(exc)
        code = 404 if message == 'unknown camera' else 409 if message == 'camera already registered' else 422
        raise HTTPException(status_code=code, detail=message) from exc

@router.get('')
def list_profiles():
    return registry().list()

@router.post('', status_code=status.HTTP_201_CREATED)
def register(payload: Registration):
    return invoke(registry().register, payload.camera_id, payload.source_type, payload.stream.model_dump(), source_location=payload.source_location)

@router.get('/{camera_id}')
def get_profile(camera_id: str):
    return invoke(registry().get, camera_id)

@router.put('/{camera_id}/stream')
def update_stream(camera_id: str, payload: Stream):
    return invoke(registry().update_stream, camera_id, payload.model_dump())

@router.patch('/{camera_id}/location')
def update_location(camera_id: str, payload: Location):
    return invoke(registry().set_location, camera_id, **payload.model_dump())

@router.patch('/{camera_id}/calibration')
def update_calibration(camera_id: str, payload: Intrinsics):
    return invoke(registry().set_calibration, camera_id, payload.model_dump())

@router.patch('/{camera_id}/orientation')
def update_orientation(camera_id: str, payload: Orientation):
    return invoke(registry().set_orientation, camera_id, **payload.model_dump())

@router.get('/{camera_id}/readiness')
def readiness(camera_id: str):
    return invoke(registry().readiness, camera_id)
