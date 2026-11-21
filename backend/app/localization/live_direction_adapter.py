"""Fail-closed adapter: image-space detections to camera-derived world directions.

This module does not produce target range, target GPS, stable tracking identity,
or evidence that a detection represents a confirmed drone. Caller declarations
of atomic-frame provenance do not authenticate the upstream acquisition path.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal

from app.localization.camera_geometry import CameraIntrinsics
from app.localization.checkerboard_calibration import CheckerboardCalibration
from app.localization.registry_optics import CameraDirectionProfile, RegistryDistortedIntrinsics
from app.localization.camera_profile import CameraGeoProfile
from app.localization.world_bearing import CameraOrientation, WorldDirection, ray_to_world_direction

FrameSource = Literal["declared_atomic_frame", "status_snapshot_unverified"]
_ALLOWED_CLASSES = frozenset(("drone", "bird", "unknown"))


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _nonnegative_index(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class DetectionBearing:
    detection_index: int  # Index in the submitted frame only, NOT stable track ID
    object_class: str  # Raw classifier output, NOT independently confirmed
    confidence: float
    center_x_px: float
    center_y_px: float
    direction: WorldDirection


@dataclass(frozen=True)
class FrameBearingResult:
    camera_id: str
    frame_index: int
    session_id: str | None
    frame_source: FrameSource  # Caller-declared provenance, not cryptographic proof
    directions: tuple[DetectionBearing, ...]


def project_frame_detections(
    frame: Mapping[str, Any],
    *,
    profile: CameraGeoProfile | CameraDirectionProfile,
    orientation: CameraOrientation,
    optics: CameraIntrinsics | CheckerboardCalibration | RegistryDistortedIntrinsics,
    frame_source: FrameSource = "status_snapshot_unverified",
) -> FrameBearingResult:
    """Project pixel-bbox centers using explicit matching camera settings.

    Required frame mapping: camera_id, frame_index, image_width_px,
    image_height_px, detections. Each detection: class, confidence,
    bbox with x/y/width/height in IMAGE PIXELS (not normalized 0..1).
    Optional per-detection camera_id/frame_index must match the frame.
    """
    if not isinstance(frame, Mapping):
        raise ValueError("frame must be a mapping")
    if not isinstance(profile, (CameraGeoProfile, CameraDirectionProfile)):
        raise ValueError("profile must be CameraGeoProfile or CameraDirectionProfile")
    if not isinstance(orientation, CameraOrientation):
        raise ValueError("orientation must be CameraOrientation")
    if frame_source not in ("declared_atomic_frame", "status_snapshot_unverified"):
        raise ValueError("Unsupported frame_source")
    if not isinstance(frame.get("camera_id"), str) or not frame["camera_id"].strip():
        raise ValueError("A nonempty frame camera_id is required")
    if frame["camera_id"] != profile.camera_id:
        raise ValueError("Frame camera_id does not match profile")
    if orientation.heading_degrees != profile.heading_degrees:
        raise ValueError("Orientation heading does not match camera profile")

    if "state" in frame and frame["state"] != "running":
        raise ValueError("Cannot project a non-running frame")

    frame_index = _nonnegative_index(frame.get("frame_index"), "frame_index")
    session_id = frame.get("session_id")
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise ValueError("session_id must be a nonempty string or None")

    width = frame.get("image_width_px")
    height = frame.get("image_height_px")
    if type(width) is not int or type(height) is not int or width < 2 or height < 2:
        raise ValueError("Explicit integer image dimensions required")

    if isinstance(optics, (CheckerboardCalibration, RegistryDistortedIntrinsics)):
        intrinsics = optics.intrinsics
        pixel_to_direction = optics.pixel_to_direction
    elif isinstance(optics, CameraIntrinsics):
        intrinsics = optics
        pixel_to_direction = optics.pixel_to_direction
    else:
        raise ValueError("Unsupported optics; measured or declared intrinsics required")

    if (width, height) != (intrinsics.width_px, intrinsics.height_px):
        raise ValueError("Frame dimensions do not match optical calibration")

    detections = frame.get("detections")
    if not isinstance(detections, (list, tuple)) or len(detections) > 512:
        raise ValueError("detections must be a list/tuple with at most 512 items")

    results: list[DetectionBearing] = []
    for i, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            raise ValueError(f"Detection {i} must be a mapping")
        label = detection.get("class")
        if not isinstance(label, str) or label not in _ALLOWED_CLASSES:
            raise ValueError(f"Detection {i}: unsupported class")
        confidence = _number(detection.get("confidence"), f"detection {i} confidence")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Detection {i}: confidence outside [0,1]")
        if "frame_index" in detection and detection["frame_index"] != frame_index:
            raise ValueError(f"Detection {i}: stale/mismatched frame_index")
        if "camera_id" in detection and detection["camera_id"] != profile.camera_id:
            raise ValueError(f"Detection {i}: mismatched camera_id")

        bbox = detection.get("bbox")
        if not isinstance(bbox, Mapping):
            raise ValueError(f"Detection {i}: bbox must be a mapping")
        x = _number(bbox.get("x"), f"detection {i} bbox.x")
        y = _number(bbox.get("y"), f"detection {i} bbox.y")
        box_width = _number(bbox.get("width"), f"detection {i} bbox.width")
        box_height = _number(bbox.get("height"), f"detection {i} bbox.height")
        if not (x >= 0 and y >= 0 and box_width > 0 and box_height > 0):
            raise ValueError(f"Detection {i}: invalid pixel bbox")
        if x + box_width > width or y + box_height > height:
            raise ValueError(f"Detection {i}: pixel bbox outside image")

        center_x = x + box_width / 2
        center_y = y + box_height / 2
        ray = pixel_to_direction(center_x, center_y)
        world = ray_to_world_direction(
            ray, orientation, optics_source=intrinsics.provenance
        )
        results.append(
            DetectionBearing(
                detection_index=i,
                object_class=label,
                confidence=confidence,
                center_x_px=center_x,
                center_y_px=center_y,
                direction=world,
            )
        )

    return FrameBearingResult(
        camera_id=profile.camera_id,
        frame_index=frame_index,
        session_id=session_id,
        frame_source=frame_source,
        directions=tuple(results),
    )
