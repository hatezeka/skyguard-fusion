"""Reproducible mathematical tests for pinhole camera geometry."""

import math

import pytest

from app.localization.camera_geometry import CameraIntrinsics


def camera():
    return CameraIntrinsics(
        width_px=1280,
        height_px=720,
        fx_px=320.0,
        fy_px=320.0,
        cx_px=640.0,
        cy_px=360.0,
        provenance="measured",
    )


def test_optical_center_points_forward():
    ray = camera().pixel_to_direction(640, 360)

    assert ray.x == pytest.approx(0)
    assert ray.y == pytest.approx(0)
    assert ray.z == pytest.approx(1)
    assert ray.yaw_offset_deg == pytest.approx(0)
    assert ray.elevation_offset_deg == pytest.approx(0)


def test_horizontal_yaw_is_geometric():
    right = camera().pixel_to_direction(960, 360)
    left = camera().pixel_to_direction(320, 360)

    assert right.yaw_offset_deg == pytest.approx(45)
    assert left.yaw_offset_deg == pytest.approx(-45)
    assert right.elevation_offset_deg == pytest.approx(0)


def test_vertical_angle_sign():
    upper = camera().pixel_to_direction(640, 200)
    lower = camera().pixel_to_direction(640, 520)

    assert upper.elevation_offset_deg > 0
    assert lower.elevation_offset_deg < 0
    assert upper.elevation_offset_deg == pytest.approx(
        -lower.elevation_offset_deg
    )


def test_off_axis_elevation_uses_full_ray_geometry():
    ray = camera().pixel_to_direction(960, 40)

    expected = math.degrees(math.atan2(1, math.sqrt(2)))
    assert ray.yaw_offset_deg == pytest.approx(45)
    assert ray.elevation_offset_deg == pytest.approx(expected)


def test_ray_is_unit_length_and_reprojects_to_pixel():
    intrinsics = camera()
    x, y = 875.25, 233.5
    ray = intrinsics.pixel_to_direction(x, y)

    assert math.hypot(ray.x, ray.y, ray.z) == pytest.approx(1)
    assert ray.z > 0

    projected_x = intrinsics.fx_px * ray.x / ray.z + intrinsics.cx_px
    projected_y = intrinsics.fy_px * ray.y / ray.z + intrinsics.cy_px

    assert projected_x == pytest.approx(x)
    assert projected_y == pytest.approx(y)


def test_nominal_fov_is_explicitly_not_measured():
    intrinsics = CameraIntrinsics.from_nominal_horizontal_fov(
        1280, 720, 90
    )

    assert intrinsics.provenance == "nominal_fov"
    assert intrinsics.fx_px == pytest.approx(640)
    assert intrinsics.fy_px == pytest.approx(640)


@pytest.mark.parametrize("fov", [0, -1, 180, 181, float("nan"), float("inf")])
def test_invalid_nominal_fov_rejected(fov):
    with pytest.raises(ValueError):
        CameraIntrinsics.from_nominal_horizontal_fov(1280, 720, fov)


@pytest.mark.parametrize(
    "coordinates",
    [
        (-1, 360),
        (1280, 360),
        (640, -1),
        (640, 720),
        (float("nan"), 360),
        (640, float("inf")),
        (True, 360),
    ],
)
def test_invalid_pixels_rejected(coordinates):
    with pytest.raises(ValueError):
        camera().pixel_to_direction(*coordinates)


@pytest.mark.parametrize(
    "changes",
    [
        {"width_px": 0},
        {"height_px": True},
        {"fx_px": 0},
        {"fy_px": -1},
        {"fx_px": float("nan")},
        {"cy_px": float("inf")},
        {"provenance": "unknown_source"},
    ],
)
def test_invalid_intrinsics_rejected(changes):
    values = {
        "width_px": 1280,
        "height_px": 720,
        "fx_px": 320.0,
        "fy_px": 320.0,
        "cx_px": 640.0,
        "cy_px": 360.0,
        "provenance": "measured",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        CameraIntrinsics(**values)


def test_no_unjustified_range_or_gps_output():
    ray = camera().pixel_to_direction(800, 300)

    assert not hasattr(ray, "distance_m")
    assert not hasattr(ray, "latitude")
    assert not hasattr(ray, "longitude")
