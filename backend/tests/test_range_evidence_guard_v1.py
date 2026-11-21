from __future__ import annotations

import math

import pytest

from app.localization.range_evidence_guard_v1 import (
    ApparentSizeEvidence, BoundedMeasurement, RangeAssessment,
    estimate_camera_range,
)


def m(nominal, lower=None, upper=None, source="reference-measurement"):
    return BoundedMeasurement(
        nominal=nominal,
        lower=nominal if lower is None else lower,
        upper=nominal if upper is None else upper,
        source=source,
    )


def ev(**changes):
    values = dict(
        focal_px=m(800, 760, 840, "independent-calibration"),
        apparent_target_size_m=m(0.4, 0.35, 0.45, "measured-target-pose"),
        measured_image_size_px=m(40, 38, 42, "bbox-image-measurement"),
        calibrated_camera=True,
        independently_sized_target=True,
        dimension_axis="width",
        frame_id="camera1-001",
        target_track_id="drone-7",
    )
    values.update(changes)
    return ApparentSizeEvidence(**values)


def test_valid_evidence_produces_optical_axis_depth_bounded_by_inputs():
    result = estimate_camera_range(ev())
    assert result.status == "estimated"
    assert result.camera_optical_axis_depth_m == pytest.approx(8.0)
    assert result.conservative_lower_m == pytest.approx(760 * .35 / 42)
    assert result.conservative_upper_m == pytest.approx(840 * .45 / 38)
    assert result.facility_distance_m is None
    assert result.latitude is None and result.longitude is None
    assert result.frame_id == "camera1-001"


def test_missing_evidence_returns_unknown_not_numeric():
    result = estimate_camera_range(None)
    assert result.status == "unknown"
    assert result.camera_optical_axis_depth_m is None


@pytest.mark.parametrize("changes,reason", [
    ({"calibrated_camera": False}, "UNVERIFIED_CAMERA_CALIBRATION"),
    ({"independently_sized_target": False}, "UNVERIFIED_EFFECTIVE_TARGET_SIZE"),
    ({"dimension_axis": ""}, "AMBIGUOUS_DIMENSION_AXIS"),
    ({"frame_id": ""}, "MISSING_FRAME_ID"),
    ({"target_track_id": ""}, "MISSING_TRACK_ID"),
    ({"focal_px": m(0)}, "INVALID_FOCAL_EVIDENCE"),
    ({"focal_px": m(float("nan"))}, "INVALID_FOCAL_EVIDENCE"),
    ({"focal_px": m(True)}, "INVALID_FOCAL_EVIDENCE"),
    ({"measured_image_size_px": m(0)}, "INVALID_IMAGE_MEASUREMENT"),
    ({"measured_image_size_px": m(10, 20, 30)}, "INVALID_IMAGE_MEASUREMENT"),
    ({"apparent_target_size_m": m(-1)}, "INVALID_TARGET_SIZE_EVIDENCE"),
    ({"apparent_target_size_m": m(.4, source="")}, "INVALID_TARGET_SIZE_EVIDENCE"),
])
def test_inadequate_evidence_is_unknown(changes, reason):
    result = estimate_camera_range(ev(**changes))
    assert result.status == "unknown" and result.reason == reason
    assert result.camera_optical_axis_depth_m is None
    assert result.conservative_lower_m is None
    assert result.conservative_upper_m is None


def test_small_image_target_does_not_invent_precision():
    # A single measured pixel cannot imply a precise distance.
    result = estimate_camera_range(ev(measured_image_size_px=m(1.0, 0.5, 2.0)))
    assert result.status == "estimated"
    assert result.conservative_upper_m > result.conservative_lower_m
    assert result.conservative_upper_m / result.conservative_lower_m > 2


def test_overflow_fails_closed():
    result = estimate_camera_range(ev(focal_px=m(1e308), apparent_target_size_m=m(1e308)))
    assert result.status == "unknown"
    assert result.reason == "NUMERIC_UNSTABLE"


def test_no_way_to_inject_facility_distance_or_gps():
    with pytest.raises(ValueError, match="facility"):
        RangeAssessment(status="estimated", reason="bad", facility_distance_m=10.0)
    with pytest.raises(ValueError, match="GPS"):
        RangeAssessment(status="estimated", reason="bad", latitude=42.0)
