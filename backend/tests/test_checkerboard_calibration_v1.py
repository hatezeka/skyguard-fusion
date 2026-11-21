"""Synthetic observations verify math, not real physical camera accuracy."""

import cv2
import numpy as np
import pytest

from app.localization.checkerboard_calibration import (
    calibrate_checkerboard_corners,
)


COLUMNS, ROWS = 8, 6
WIDTH, HEIGHT = 1280, 720
SQUARE_SIZE = 0.035

KNOWN_K = np.array(
    [[940.0, 0, 630.0], [0, 925.0, 355.0], [0, 0, 1.0]],
    dtype=np.float64,
)

KNOWN_DISTORTION = np.array(
    [0.016, -0.006, 0.001, -0.0008, 0.001],
    dtype=np.float64,
)


def observations():
    board = np.zeros(
        (ROWS * COLUMNS, 3),
        dtype=np.float32,
    )
    board[:, :2] = np.mgrid[
        0:COLUMNS, 0:ROWS
    ].T.reshape(-1, 2)
    board[:, :2] *= SQUARE_SIZE

    result = []

    for index in range(12):
        rotation = np.array([
            0.04 + (index % 4) * 0.075,
            -0.18 + (index % 5) * 0.083,
            -0.12 + (index % 3) * 0.06,
        ])

        translation = np.array([
            -0.20 + (index % 5) * 0.075,
            -0.13 + (index % 4) * 0.035,
            0.8 + (index % 4) * 0.095,
        ])

        projected, _ = cv2.projectPoints(
            board,
            rotation,
            translation,
            KNOWN_K,
            KNOWN_DISTORTION,
        )

        result.append(projected.reshape(-1, 2).tolist())

    return result


def calibrate(**options):
    return calibrate_checkerboard_corners(
        observations(),
        image_width=WIDTH,
        image_height=HEIGHT,
        board_columns=COLUMNS,
        board_rows=ROWS,
        square_size=SQUARE_SIZE,
        **options,
    )


def test_recovers_intrinsic_parameters():
    result = calibrate(max_rms_px=0.2)

    assert result.views == 12
    assert result.rms_px < 0.2
    assert max(result.per_view_rmse_px) < 0.2
    assert result.intrinsics.provenance == "measured"

    assert result.intrinsics.fx_px == pytest.approx(940, abs=12)
    assert result.intrinsics.fy_px == pytest.approx(925, abs=12)
    assert result.intrinsics.cx_px == pytest.approx(630, abs=15)
    assert result.intrinsics.cy_px == pytest.approx(355, abs=15)


def test_undistorted_ray_matches_known_direction():
    result = calibrate()

    point = np.array([[0.10, 0.06, 1.0]])

    pixel, _ = cv2.projectPoints(
        point,
        np.zeros(3),
        np.zeros(3),
        KNOWN_K,
        KNOWN_DISTORTION,
    )

    x, y = pixel.reshape(2)

    ray = result.pixel_to_direction(float(x), float(y))

    assert ray.x / ray.z == pytest.approx(0.10, abs=0.006)
    assert ray.y / ray.z == pytest.approx(0.06, abs=0.006)
    assert not hasattr(ray, "distance_m")


def test_identical_views_are_rejected():
    first = observations()[0]
    repeated = [first] * 7

    with pytest.raises(ValueError, match="diversity"):
        calibrate_checkerboard_corners(
            repeated,
            image_width=WIDTH,
            image_height=HEIGHT,
            board_columns=COLUMNS,
            board_rows=ROWS,
            square_size=SQUARE_SIZE,
        )


def test_insufficient_views_are_rejected():
    with pytest.raises(ValueError):
        calibrate_checkerboard_corners(
            observations()[:4],
            image_width=WIDTH,
            image_height=HEIGHT,
            board_columns=COLUMNS,
            board_rows=ROWS,
            square_size=SQUARE_SIZE,
        )


def test_invalid_square_size_is_rejected():
    with pytest.raises(ValueError):
        calibrate_checkerboard_corners(
            observations(),
            image_width=WIDTH,
            image_height=HEIGHT,
            board_columns=COLUMNS,
            board_rows=ROWS,
            square_size=-1,
        )


def test_excessive_reprojection_error_is_rejected():
    with pytest.raises(ValueError, match="RMS"):
        calibrate(max_rms_px=1e-12)


def test_nonfinite_observations_are_rejected():
    views = observations()
    views[0][0][0] = float("nan")

    with pytest.raises(ValueError, match="nonfinite"):
        calibrate_checkerboard_corners(
            views,
            image_width=WIDTH,
            image_height=HEIGHT,
            board_columns=COLUMNS,
            board_rows=ROWS,
            square_size=SQUARE_SIZE,
        )
