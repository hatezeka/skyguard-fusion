"""Registered-camera frame directions; no range, target GPS, or proof of survey.

The registry is read when the request is processed. A mutable registry is NOT
proof that its values were the active calibration when the JPEG was captured.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, StrictInt, StrictStr

from app.localization.camera_geometry import CameraIntrinsics
from app.localization.live_direction_adapter import project_frame_detections
from app.localization.registry_optics import CameraDirectionProfile, RegistryDistortedIntrinsics
from app.localization.world_bearing import CameraOrientation


class RegisteredBearingRequest(BaseModel):
    camera_id: StrictStr = Field(
        min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    expected_frame_index: StrictInt = Field(ge=0)
    expected_session_id: StrictStr = Field(
        min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"
    )
    # Registry contains no north-reference field, so this stays explicit.
    north_reference: Literal["true_north", "magnetic_north"]


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_registry(
    record: Mapping, camera_id: str, width: int, height: int
) -> tuple[CameraDirectionProfile, RegistryDistortedIntrinsics, tuple[float, float, float], str]:
    if not isinstance(record, Mapping) or record.get("camera_id") != camera_id:
        raise ValueError("Registry camera_id does not match live frame")
    stream = record.get("stream")
    if not isinstance(stream, Mapping) or (
        type(stream.get("width_px")) is not int
        or type(stream.get("height_px")) is not int
        or (stream["width_px"], stream["height_px"]) != (width, height)
    ):
        raise ValueError("Registry stream resolution does not match JPEG")

    cal = record.get("calibration")
    if not isinstance(cal, Mapping) or cal.get("status") != "CALIBRATED":
        raise ValueError("Registry calibration is not CALIBRATED")
    if cal.get("resolution") != [width, height]:
        raise ValueError("Registry calibration resolution does not match JPEG")
    if not isinstance(cal.get("verified_at"), str) or not cal["verified_at"].strip():
        raise ValueError("Registry calibration timestamp is missing")
    values = cal.get("intrinsics")
    if not isinstance(values, Mapping):
        raise ValueError("Registry camera intrinsics are missing")
    fx, fy, cx, cy = (
        _finite(values.get(key), f"intrinsics.{key}")
        for key in ("fx", "fy", "cx", "cy")
    )
    if fx <= 0 or fy <= 0 or not (0 <= cx < width and 0 <= cy < height):
        raise ValueError("Invalid registered optical intrinsics")
    dist = values.get("distortion")
    if not isinstance(dist, (tuple, list)) or len(dist) != 5:
        raise ValueError("Five registry distortion coefficients required")
    distortion = tuple(_finite(value, "distortion") for value in dist)
    optics = RegistryDistortedIntrinsics(
        intrinsics=CameraIntrinsics(
            width_px=width,
            height_px=height,
            fx_px=fx,
            fy_px=fy,
            cx_px=cx,
            cy_px=cy,
            # Manual registration cannot independently prove measured provenance.
            provenance="unverified",
        ),
        distortion=distortion,
    )

    pose = record.get("orientation")
    if not isinstance(pose, Mapping):
        raise ValueError("Registry orientation is missing")
    source = pose.get("source")
    if source not in ("sensor", "manual", "calibration", "simulation"):
        raise ValueError("Registry orientation source is not set")
    yaw = _finite(pose.get("yaw_deg"), "orientation.yaw_deg")
    pitch = _finite(pose.get("pitch_deg"), "orientation.pitch_deg")
    roll = _finite(pose.get("roll_deg"), "orientation.roll_deg")
    # The north reference is in the request; registry does not persist it.
    # Regardless of source, it is not an independently surveyed orientation.
    profile = CameraDirectionProfile(camera_id=camera_id, heading_degrees=yaw)
    return profile, optics, (yaw, pitch, roll), source


def derive_registered_frame_bearings(
    snapshot: object,
    jpeg: bytes,
    *,
    request: RegisteredBearingRequest,
    registry_record: Mapping,
) -> dict:
    """Derive bearings strictly from one atomic frame and current registry state."""
    if snapshot is None or getattr(snapshot, "state", None) != "running":
        raise ValueError("Live frame is not running")
    if getattr(snapshot, "camera_id", None) != request.camera_id:
        raise ValueError("Active camera_id does not match request")
    frame_index = getattr(snapshot, "frame_index", None)
    if type(frame_index) is not int or frame_index != request.expected_frame_index:
        raise ValueError("Active frame_index does not match request")
    session_id = getattr(snapshot, "session_id", None)
    if not isinstance(session_id, str) or session_id != request.expected_session_id:
        raise ValueError("Active session_id does not match request")

    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= 16 * 1024 * 1024:
        raise ValueError("Synchronized JPEG is unavailable or too large")
    try:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError("Synchronized JPEG cannot be decoded") from exc
    if image is None or image.ndim not in (2, 3):
        raise ValueError("Synchronized JPEG cannot be decoded")
    height, width = image.shape[:2]
    if not (2 <= width <= 8192 and 2 <= height <= 8192 and width * height <= 24_000_000):
        raise ValueError("JPEG dimensions exceed supported limits")

    profile, optics, pose, source = _validate_registry(
        registry_record, request.camera_id, width, height
    )
    orientation = CameraOrientation(
        heading_degrees=pose[0],
        pitch_degrees=pose[1],
        roll_degrees=pose[2],
        north_reference=request.north_reference,
        orientation_source="manual_unverified",
    )
    result = project_frame_detections(
        {
            "state": "running",
            "camera_id": request.camera_id,
            "frame_index": frame_index,
            "session_id": session_id,
            "image_width_px": width,
            "image_height_px": height,
            "detections": getattr(snapshot, "detections", None),
        },
        profile=profile,
        orientation=orientation,
        optics=optics,
        frame_source="declared_atomic_frame",
    )
    return {
        "camera_id": request.camera_id,
        "session_id": session_id,
        "frame_index": frame_index,
        "image_width_px": width,
        "image_height_px": height,
        "directions": [asdict(item) for item in result.directions],
        "frame_source": "manager.latest_frame_atomic",
        "profile_source": "camera_registry_current",
        "registry_calibration_status": "CALIBRATED",
        "optics_source": "registry_intrinsics_unverified",
        "orientation_source": f"registry_{source}_unverified",
        "north_reference": request.north_reference,
        "session_identity_verified": True,
        "profile_bound_to_session_verified": False,
        "profile_bound_to_frame_verified": False,
        "physical_calibration_verified": False,
        "physical_distance_available": False,
        "target_gps_available": False,
    }
