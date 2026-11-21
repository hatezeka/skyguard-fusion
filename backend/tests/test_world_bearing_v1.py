"""Deterministic geometry and safety tests for camera-to-ENU rotation."""
import math

import pytest

from app.localization.camera_geometry import CameraIntrinsics, PixelDirection
from app.localization.world_bearing import (
    CameraOrientation,
    ray_to_world_direction,
)


def camera():
    return CameraIntrinsics(
        width_px=1280, height_px=720,
        fx_px=320.0, fy_px=320.0,
        cx_px=640.0, cy_px=360.0,
        provenance="measured",
    )


def world(x, y, *, heading=0.0, pitch=0.0, roll=0.0,
          north_reference="true_north", orientation_source="surveyed"):
    ray = camera().pixel_to_direction(x, y)
    return ray_to_world_direction(
        ray,
        CameraOrientation(
            heading_degrees=heading,
            north_reference=north_reference,
            pitch_degrees=pitch,
            roll_degrees=roll,
            orientation_source=orientation_source,
        ),
        optics_source="measured",
    )


def test_center_facing_north():
    result = world(640, 360)
    assert result.azimuth_degrees == pytest.approx(0)
    assert result.elevation_degrees == pytest.approx(0)
    assert (result.east, result.north, result.up) == pytest.approx((0, 1, 0))


def test_center_facing_east():
    result = world(640, 360, heading=90)
    assert result.azimuth_degrees == pytest.approx(90)
    assert (result.east, result.north, result.up) == pytest.approx((1, 0, 0))


def test_right_pixel_increases_clockwise_bearing():
    assert world(960, 360).azimuth_degrees == pytest.approx(45)
    assert world(320, 360).azimuth_degrees == pytest.approx(315)
    assert world(960, 360, heading=90).azimuth_degrees == pytest.approx(135)


def test_camera_pitch_controls_center_elevation():
    assert world(640, 360, pitch=30).elevation_degrees == pytest.approx(30)
    assert world(640, 360, pitch=-20).elevation_degrees == pytest.approx(-20)


def test_roll_rotates_image_right_toward_ground():
    right = world(960, 360, roll=90)
    assert right.azimuth_degrees == pytest.approx(0)
    assert right.elevation_degrees == pytest.approx(-45)


def test_compound_rotation_preserves_unit_vector():
    result = world(725, 390, heading=123, pitch=18, roll=-33)
    assert math.hypot(result.east, result.north, result.up) == pytest.approx(1)
    assert -90 <= result.elevation_degrees <= 90
    assert result.azimuth_degrees is not None
    assert 0 <= result.azimuth_degrees < 360


def test_vertical_ray_has_no_azimuth():
    result = world(640, 360, pitch=90)
    assert result.azimuth_degrees is None
    assert result.elevation_degrees == pytest.approx(90)


def test_magnetic_reference_is_not_misreported_as_true_north():
    result = world(640, 360, heading=90, north_reference="magnetic_north")
    assert result.north_reference == "magnetic_north"
    assert result.azimuth_degrees == pytest.approx(90)
    assert not hasattr(result, "true_north_azimuth_degrees")


def test_provisional_orientation_remains_labeled():
    result = world(640, 360, orientation_source="manual_unverified")
    assert result.orientation_source == "manual_unverified"
    assert result.optics_source == "measured"
    assert not hasattr(result, "distance_m")
    assert not hasattr(result, "latitude")
    assert not hasattr(result, "longitude")


def test_nominal_optics_stays_explicit():
    intr = CameraIntrinsics.from_nominal_horizontal_fov(1280, 720, 90)
    result = ray_to_world_direction(
        intr.pixel_to_direction(900, 300),
        CameraOrientation(10, "true_north"),
        optics_source=intr.provenance,
    )
    assert result.optics_source == "nominal_fov"
    assert result.orientation_source == "manual_unverified"


@pytest.mark.parametrize("kwargs", [
    {"heading_degrees": 360},
    {"heading_degrees": -1},
    {"heading_degrees": float("nan")},
    {"heading_degrees": True},
    {"pitch_degrees": -91},
    {"pitch_degrees": float("inf")},
    {"roll_degrees": float("nan")},
    {"north_reference": "unknown"},
    {"north_reference": None},
    {"orientation_source": "unknown"},
])
def test_invalid_orientation_is_rejected(kwargs):
    args = {"heading_degrees": 0, "north_reference": "true_north"}
    args.update(kwargs)
    with pytest.raises(ValueError):
        CameraOrientation(**args)


@pytest.mark.parametrize("components", [
    (0, 0, 0), (0, 0, -1),
    (float("nan"), 0, 1), (0, float("inf"), 1),
    (True, 0, 1),
])
def test_invalid_camera_rays_rejected(components):
    ray = PixelDirection(*components, yaw_offset_deg=0, elevation_offset_deg=0)
    with pytest.raises(ValueError):
        ray_to_world_direction(ray, CameraOrientation(0, "true_north"))


def test_invalid_optics_provenance_rejected():
    with pytest.raises(ValueError, match="optics_source"):
        ray_to_world_direction(
            camera().pixel_to_direction(640, 360),
            CameraOrientation(0, "true_north"),
            optics_source="trusted_with_no_evidence",
        )
