"""Evidence-gated monocular range estimate for SkyGuard FUSION.

A bearing/pixel coordinate is not a range. Only report an optical-axis
depth estimate if independently evidenced camera intrinsics, target apparent dimension,
and pixel measurement are provided, with bounded intervals.

This module deliberately DOES NOT produce facility-relative distance or GPS.
It is a conservative, dependency-free guard; it does not replace the existing
localization pipeline. Real calibration/field validation is reserved for phase 45.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class BoundedMeasurement:
    """Nominal value and explicit conservative bounds in one unit."""
    nominal: float
    lower: float
    upper: float
    source: str


@dataclass(frozen=True)
class ApparentSizeEvidence:
    focal_px: BoundedMeasurement
    apparent_target_size_m: BoundedMeasurement
    measured_image_size_px: BoundedMeasurement
    calibrated_camera: bool
    independently_sized_target: bool
    frame_id: str
    target_track_id: str
    # Determines what physical/apparent dimension was measured. Bounding box
    # width cannot blindly be paired with an object's catalog width.
    dimension_axis: str = ""


@dataclass(frozen=True)
class RangeAssessment:
    status: str  # 'unknown' or 'estimated'
    reason: str
    camera_optical_axis_depth_m: Optional[float] = None
    conservative_lower_m: Optional[float] = None
    conservative_upper_m: Optional[float] = None
    method: str = "NONE"
    frame_id: Optional[str] = None
    target_track_id: Optional[str] = None
    facility_distance_m: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def __post_init__(self) -> None:
        if self.facility_distance_m is not None or self.latitude is not None or self.longitude is not None:
            raise ValueError("Monocular apparent-size estimates must never imply GPS or facility-relative distance")


def _measurement_valid(m: BoundedMeasurement | None) -> bool:
    if not isinstance(m, BoundedMeasurement) or not isinstance(m.source, str) or not m.source.strip():
        return False
    for value in (m.nominal, m.lower, m.upper):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if not isfinite(value) or value <= 0:
            return False
    return m.lower <= m.nominal <= m.upper


def estimate_camera_range(evidence: ApparentSizeEvidence | None) -> RangeAssessment:
    """Produce cautious camera optical-axis depth with explicit bound assumptions.

    Pinhole approximation: Z = f * S / p, where f is calibrated focal length
    in pixels, S is effective apparent target dimension in metres for the
    observed pose, and p is measured corresponding image dimension in pixels.
    Declared lower/upper input intervals produce deterministic extrema; these
    are *not* calibrated 95% confidence intervals.

    Unknown is deliberately preferable to a fabricated numeric distance.
    """
    def unknown(reason: str, ev: ApparentSizeEvidence | None = None) -> RangeAssessment:
        return RangeAssessment(status="unknown", reason=reason,
                               frame_id=getattr(ev, "frame_id", None),
                               target_track_id=getattr(ev, "target_track_id", None))

    if not isinstance(evidence, ApparentSizeEvidence):
        return unknown("MISSING_EVIDENCE")
    if evidence.calibrated_camera is not True:
        return unknown("UNVERIFIED_CAMERA_CALIBRATION", evidence)
    if evidence.independently_sized_target is not True:
        return unknown("UNVERIFIED_EFFECTIVE_TARGET_SIZE", evidence)
    if not isinstance(evidence.frame_id, str) or not evidence.frame_id.strip():
        return unknown("MISSING_FRAME_ID", evidence)
    if not isinstance(evidence.target_track_id, str) or not evidence.target_track_id.strip():
        return unknown("MISSING_TRACK_ID", evidence)
    if evidence.dimension_axis not in ("width", "height"):
        return unknown("AMBIGUOUS_DIMENSION_AXIS", evidence)
    if not _measurement_valid(evidence.focal_px):
        return unknown("INVALID_FOCAL_EVIDENCE", evidence)
    if not _measurement_valid(evidence.apparent_target_size_m):
        return unknown("INVALID_TARGET_SIZE_EVIDENCE", evidence)
    if not _measurement_valid(evidence.measured_image_size_px):
        return unknown("INVALID_IMAGE_MEASUREMENT", evidence)

    f = evidence.focal_px
    size = evidence.apparent_target_size_m
    pixel = evidence.measured_image_size_px
    estimate = f.nominal * size.nominal / pixel.nominal
    lower = f.lower * size.lower / pixel.upper
    upper = f.upper * size.upper / pixel.lower
    if not all(isfinite(v) and v > 0 for v in (estimate, lower, upper)) or not lower <= estimate <= upper:
        return unknown("NUMERIC_UNSTABLE", evidence)

    return RangeAssessment(
        status="estimated",
        reason="CAMERA_RANGE_ONLY_APPROXIMATE_NOT_PHYSICALLY_VALIDATED",
        camera_optical_axis_depth_m=estimate,
        conservative_lower_m=lower,
        conservative_upper_m=upper,
        method="CALIBRATED_MONOCULAR_APPARENT_SIZE_PINHOLE",
        frame_id=evidence.frame_id,
        target_track_id=evidence.target_track_id,
    )
