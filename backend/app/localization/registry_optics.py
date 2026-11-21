"""Registry-backed optical ray geometry without invented camera GPS.

The registry's CALIBRATED label is metadata, not proof of external accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from app.localization.camera_geometry import CameraIntrinsics, PixelDirection


@dataclass(frozen=True)
class CameraDirectionProfile:
    """Only the camera identity and heading needed for ray projection."""

    camera_id: str
    heading_degrees: float

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id is required")
        if type(self.heading_degrees) not in (int, float) or not math.isfinite(self.heading_degrees):
            raise ValueError("Invalid camera heading")
        if not 0 <= self.heading_degrees < 360:
            raise ValueError("Camera heading outside [0, 360)")


@dataclass(frozen=True)
class RegistryDistortedIntrinsics:
    """Undistort a pixel using the registry's 5-coefficient OpenCV lens model."""

    intrinsics: CameraIntrinsics
    distortion: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise ValueError("Valid intrinsics are required")
        if len(self.distortion) != 5 or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in self.distortion
        ):
            raise ValueError("Five finite distortion coefficients are required")

    def pixel_to_direction(self, x_px: float, y_px: float) -> PixelDirection:
        # Shared pixel and image-bound validation.
        self.intrinsics.pixel_to_direction(x_px, y_px)
        k = self.intrinsics
        matrix = np.array(
            [[k.fx_px, 0, k.cx_px], [0, k.fy_px, k.cy_px], [0, 0, 1]],
            dtype=np.float64,
        )
        pixel = np.array([[[float(x_px), float(y_px)]]], dtype=np.float64)
        try:
            coords = cv2.undistortPoints(
                pixel, matrix, np.asarray(self.distortion, dtype=np.float64)
            )[0, 0]
        except cv2.error as exc:
            raise ValueError("OpenCV could not undistort pixel") from exc
        nx, ny = float(coords[0]), float(coords[1])
        if not math.isfinite(nx) or not math.isfinite(ny):
            raise ValueError("Lens undistortion produced nonfinite ray")
        norm = math.hypot(nx, ny, 1.0)
        return PixelDirection(
            x=nx / norm,
            y=ny / norm,
            z=1.0 / norm,
            yaw_offset_deg=math.degrees(math.atan2(nx, 1.0)),
            elevation_offset_deg=math.degrees(math.atan2(-ny, math.hypot(nx, 1.0))),
        )
