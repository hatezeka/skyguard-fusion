"""Camera geospatial profiles for SkyGuard FUSION."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Literal

from app.localization.coordinates import destination_point

PositionSource = Literal["manual", "gnss"]


@dataclass(frozen=True)
class CameraGeoProfile:
    camera_id: str
    latitude: float
    longitude: float
    heading_degrees: float
    horizontal_fov_degrees: float
    coverage_radius_m: float
    position_source: PositionSource = "manual"
    position_accuracy_m: float | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id cannot be empty")

        numeric = (
            self.latitude,
            self.longitude,
            self.heading_degrees,
            self.horizontal_fov_degrees,
            self.coverage_radius_m,
        )

        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("Camera numeric values must be finite")

        if not -90 <= self.latitude <= 90:
            raise ValueError("Invalid latitude")

        if not -180 <= self.longitude <= 180:
            raise ValueError("Invalid longitude")

        if not 0 <= self.heading_degrees < 360:
            raise ValueError("Heading must be in [0, 360)")

        if not 0 < self.horizontal_fov_degrees <= 180:
            raise ValueError("Horizontal FOV must be in (0, 180]")

        if self.coverage_radius_m <= 0:
            raise ValueError("Coverage radius must be positive")

        if self.position_source not in ("manual", "gnss"):
            raise ValueError("Unsupported position source")

        if self.position_accuracy_m is not None:
            if (
                not math.isfinite(self.position_accuracy_m)
                or self.position_accuracy_m < 0
            ):
                raise ValueError("Invalid position accuracy")

        if self.updated_at is not None:
            if (
                self.updated_at.tzinfo is None
                or self.updated_at.utcoffset() is None
            ):
                raise ValueError("updated_at must include timezone")

    def coverage_polygon(
        self,
        segments: int = 32,
    ) -> list[tuple[float, float]]:
        """Return theoretical FOV sector in (latitude, longitude) order."""

        if not isinstance(segments, int) or segments < 2:
            raise ValueError("segments must be an integer >= 2")

        start = self.heading_degrees - self.horizontal_fov_degrees / 2
        step = self.horizontal_fov_degrees / segments

        polygon = [(self.latitude, self.longitude)]

        for index in range(segments + 1):
            bearing = start + index * step

            polygon.append(
                destination_point(
                    self.latitude,
                    self.longitude,
                    bearing,
                    self.coverage_radius_m,
                )
            )

        polygon.append((self.latitude, self.longitude))

        return polygon


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
