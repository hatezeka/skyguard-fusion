"""Fail-closed live bearing calculation from an atomic snapshot/JPEG pair.

Does not estimate target GPS/range or verify real-world camera calibration.
A configured geo profile is used for heading/FOV only; pitch, roll and
north reference are *caller declarations* and remain unverified.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, StrictInt, StrictStr

from app.localization.camera_geometry import CameraIntrinsics
from app.localization.camera_profile import CameraGeoProfile
from app.localization.live_direction_adapter import project_frame_detections
from app.localization.world_bearing import CameraOrientation


class LiveBearingRequest(BaseModel):
    camera_id: StrictStr = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    expected_frame_index: StrictInt = Field(ge=0)
    expected_session_id: StrictStr = Field(
        min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$"
    )
    north_reference: Literal["true_north", "magnetic_north"]
    pitch_degrees: float = Field(ge=-90, le=90, allow_inf_nan=False)
    roll_degrees: float = Field(ge=-180, le=180, allow_inf_nan=False)


def derive_live_frame_bearings(
    snapshot: object,
    jpeg: bytes,
    *,
    request: LiveBearingRequest,
    profile: CameraGeoProfile,
) -> dict:
    """Project same-frame detection bboxes without a second live status read.

    A frame index alone cannot distinguish two restarted sessions; the
    response explicitly identifies that limitation. The caller must not use
    it as verified track/session identity.
    """
    if snapshot is None or getattr(snapshot, "state", None) != "running":
        raise ValueError("Live frame is not running")
    camera_id = getattr(snapshot, "camera_id", None)
    frame_index = getattr(snapshot, "frame_index", None)
    if not isinstance(camera_id, str) or camera_id != request.camera_id:
        raise ValueError("Active camera does not match requested camera_id")
    if type(frame_index) is not int or frame_index != request.expected_frame_index:
        raise ValueError("Active frame_index does not match requested frame")
    if camera_id != profile.camera_id:
        raise ValueError("Geo profile camera_id does not match active camera")
    session_id = getattr(snapshot, "session_id", None)
    if (
        not isinstance(session_id, str)
        or session_id != request.expected_session_id
    ):
        raise ValueError("Active session_id does not match requested session")

    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= 16 * 1024 * 1024:
        raise ValueError("Synchronized JPEG is unavailable or too large")
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3):
        raise ValueError("Synchronized JPEG is not a decodable image")
    height, width = image.shape[:2]
    if not (2 <= width <= 8192 and 2 <= height <= 8192 and width * height <= 24_000_000):
        raise ValueError("Image dimensions exceed supported limits")

    detections = getattr(snapshot, "detections", None)
    optics = CameraIntrinsics.from_nominal_horizontal_fov(
        width, height, profile.horizontal_fov_degrees
    )
    orientation = CameraOrientation(
        heading_degrees=profile.heading_degrees,
        north_reference=request.north_reference,
        pitch_degrees=request.pitch_degrees,
        roll_degrees=request.roll_degrees,
        orientation_source="manual_unverified",
    )
    result = project_frame_detections(
        {
            "camera_id": camera_id,
            "frame_index": frame_index,
            "state": "running",
            "image_width_px": width,
            "image_height_px": height,
            "detections": detections,
        },
        profile=profile,
        orientation=orientation,
        optics=optics,
        frame_source="declared_atomic_frame",
    )
    return {
        "camera_id": camera_id,
        "frame_index": frame_index,
        "session_id": session_id,
        "image_width_px": width,
        "image_height_px": height,
        "directions": [asdict(d) for d in result.directions],
        "frame_source": "manager.latest_frame_atomic",
        "optics_source": "nominal_fov",
        "orientation_source": "manual_unverified",
        "north_reference": request.north_reference,
        "session_identity_verified": True,
        "profile_bound_to_frame_verified": False,
        "physical_distance_available": False,
        "target_gps_available": False,
    }
