"""Pinhole camera geometry for SkyGuard FUSION.

Coordinates: camera X right, Y down, Z forward.

This module estimates directions, never object range or GPS.
Nominal FOV parameters are assumptions, not measured calibration.
Lens distortion and camera-to-world rotation are not modeled here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


Provenance = Literal["measured", "nominal_fov", "unverified"]


def _finite_real(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class PixelDirection:
    """Unit ray and angular offsets relative to the optical axis."""

    x: float
    y: float
    z: float
    yaw_offset_deg: float
    elevation_offset_deg: float


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    provenance: Provenance = "unverified"

    def __post_init__(self) -> None:
        if (
            type(self.width_px) is not int
            or type(self.height_px) is not int
            or self.width_px < 2
            or self.height_px < 2
        ):
            raise ValueError("Image dimensions must be integers >= 2")

        for name in ("fx_px", "fy_px", "cx_px", "cy_px"):
            _finite_real(getattr(self, name), name)

        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError("Focal lengths must be positive")

        if self.provenance not in (
            "measured", "nominal_fov", "unverified"
        ):
            raise ValueError("Unsupported calibration provenance")

    @classmethod
    def from_nominal_horizontal_fov(
        cls,
        width_px: int,
        height_px: int,
        horizontal_fov_deg: float,
    ) -> "CameraIntrinsics":
        """Approximation: centered principal point, square pixels.

        The FOV alone is not a full lens calibration.
        """

        if (
            type(width_px) is not int
            or type(height_px) is not int
            or width_px < 2
            or height_px < 2
        ):
            raise ValueError("Invalid image dimensions")

        fov = _finite_real(horizontal_fov_deg, "horizontal_fov_deg")
        if not 0 < fov < 180:
            raise ValueError("Horizontal FOV must be in (0, 180)")

        focal = width_px / (2 * math.tan(math.radians(fov) / 2))

        return cls(
            width_px=width_px,
            height_px=height_px,
            fx_px=focal,
            fy_px=focal,
            cx_px=width_px / 2,
            cy_px=height_px / 2,
            provenance="nominal_fov",
        )

    def pixel_to_direction(
        self,
        x_px: float,
        y_px: float,
    ) -> PixelDirection:
        """Convert a pixel center into a normalized camera-space ray."""

        x = _finite_real(x_px, "x_px")
        y = _finite_real(y_px, "y_px")

        if not (0 <= x < self.width_px and 0 <= y < self.height_px):
            raise ValueError("Pixel coordinates outside image")

        nx = (x - self.cx_px) / self.fx_px
        ny = (y - self.cy_px) / self.fy_px

        if not (math.isfinite(nx) and math.isfinite(ny)):
            raise ValueError("Unrepresentable normalized coordinates")

        length = math.hypot(nx, ny, 1.0)

        yaw = math.degrees(math.atan2(nx, 1.0))
        elevation = math.degrees(
            math.atan2(-ny, math.hypot(nx, 1.0))
        )

        return PixelDirection(
            x=nx / length,
            y=ny / length,
            z=1.0 / length,
            yaw_offset_deg=yaw,
            elevation_offset_deg=elevation,
        )
