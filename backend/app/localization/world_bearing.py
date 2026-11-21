"""Camera-relative rays expressed as horizon-aligned azimuth/elevation.

Coordinate systems:
* Camera: +X image right, +Y image down, +Z optical axis forward.
* Local ENU: +E east, +N north, +U vertically up.
* Heading: clockwise from the explicitly declared north reference.
* Pitch: positive when the optical axis points above the horizon.
* Roll: positive when camera image-right tilts toward the ground.

This module cannot infer true north, physical distance, target GPS,
heading uncertainty, or calibration provenance. Inputs must be declared.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from app.localization.camera_geometry import PixelDirection

NorthReference = Literal["true_north", "magnetic_north"]
OrientationSource = Literal["surveyed", "manual_unverified"]
OpticsSource = Literal["measured", "nominal_fov", "unverified"]


def _finite(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class CameraOrientation:
    heading_degrees: float
    north_reference: NorthReference
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    orientation_source: OrientationSource = "manual_unverified"

    def __post_init__(self) -> None:
        h = _finite(self.heading_degrees, "heading_degrees")
        p = _finite(self.pitch_degrees, "pitch_degrees")
        r = _finite(self.roll_degrees, "roll_degrees")
        if not 0 <= h < 360:
            raise ValueError("heading_degrees must be in [0, 360)")
        if not -90 <= p <= 90:
            raise ValueError("pitch_degrees must be in [-90, 90]")
        if not -180 <= r <= 180:
            raise ValueError("roll_degrees must be in [-180, 180]")
        if self.north_reference not in ("true_north", "magnetic_north"):
            raise ValueError("Explicit true_north/magnetic_north reference required")
        if self.orientation_source not in ("surveyed", "manual_unverified"):
            raise ValueError("Invalid orientation_source")


@dataclass(frozen=True)
class WorldDirection:
    east: float
    north: float
    up: float
    azimuth_degrees: float | None
    elevation_degrees: float
    north_reference: NorthReference
    orientation_source: OrientationSource
    optics_source: OpticsSource


def ray_to_world_direction(
    ray: PixelDirection,
    orientation: CameraOrientation,
    *,
    optics_source: OpticsSource = "unverified",
) -> WorldDirection:
    """Rotate a unit camera ray into ENU, without estimating target range.

    azimuth_degrees is relative to the declared north reference. It is
    undefined for a precisely vertical ray (returned as None).
    """
    if optics_source not in ("measured", "nominal_fov", "unverified"):
        raise ValueError("Invalid optics_source")
    x = _finite(ray.x, "ray.x")
    y = _finite(ray.y, "ray.y")
    z = _finite(ray.z, "ray.z")
    if z <= 0:
        raise ValueError("Ray must point in front of the camera")
    size = math.hypot(x, y, z)
    if not math.isfinite(size) or size == 0:
        raise ValueError("Invalid camera ray magnitude")
    x, y, z = x / size, y / size, z / size

    heading = math.radians(orientation.heading_degrees)
    pitch = math.radians(orientation.pitch_degrees)
    roll = math.radians(orientation.roll_degrees)

    sh, ch = math.sin(heading), math.cos(heading)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)

    # Orthogonal basis in east/north/up coordinates at zero roll.
    right = (ch, -sh, 0.0)
    down = (sh * sp, ch * sp, -cp)
    forward = (sh * cp, ch * cp, sp)

    # Positive roll tilts camera image-right toward local down.
    right_rolled = tuple(cr * right[i] + sr * down[i] for i in range(3))
    down_rolled = tuple(-sr * right[i] + cr * down[i] for i in range(3))

    east, north, up = (
        x * right_rolled[i] + y * down_rolled[i] + z * forward[i]
        for i in range(3)
    )
    magnitude = math.hypot(east, north, up)
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Invalid world ray")
    east, north, up = east / magnitude, north / magnitude, up / magnitude

    horizontal = math.hypot(east, north)
    azimuth = (
        math.degrees(math.atan2(east, north)) % 360.0
        if horizontal > 1e-12 else None
    )
    elevation = math.degrees(math.atan2(up, horizontal))

    return WorldDirection(
        east=east,
        north=north,
        up=up,
        azimuth_degrees=azimuth,
        elevation_degrees=elevation,
        north_reference=orientation.north_reference,
        orientation_source=orientation.orientation_source,
        optics_source=optics_source,
    )
