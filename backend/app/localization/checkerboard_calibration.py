"""Offline pinhole calibration using checkerboard corner observations.

Produces camera-relative rays only. No GPS, target range or world orientation.
Actual physical calibration requires real, diverse checkerboard photographs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import cv2
import numpy as np

from app.localization.camera_geometry import CameraIntrinsics, PixelDirection


@dataclass(frozen=True)
class CheckerboardCalibration:
    intrinsics: CameraIntrinsics
    distortion: tuple[float, ...]  # k1, k2, p1, p2, k3
    rms_px: float
    per_view_rmse_px: tuple[float, ...]
    views: int
    board_columns: int
    board_rows: int

    def pixel_to_direction(
        self,
        x_px: float,
        y_px: float,
    ) -> PixelDirection:
        """Undistort an observed pixel and return a unit camera-space ray."""

        self.intrinsics.pixel_to_direction(x_px, y_px)

        camera_matrix = _camera_matrix(self.intrinsics)
        point = np.array(
            [[[float(x_px), float(y_px)]]],
            dtype=np.float64,
        )

        undistorted = cv2.undistortPoints(
            point,
            camera_matrix,
            np.asarray(self.distortion, dtype=np.float64),
        )[0, 0]

        nx = float(undistorted[0])
        ny = float(undistorted[1])

        if not math.isfinite(nx) or not math.isfinite(ny):
            raise ValueError("Undistortion returned invalid coordinates")

        magnitude = math.hypot(nx, ny, 1.0)

        return PixelDirection(
            x=nx / magnitude,
            y=ny / magnitude,
            z=1.0 / magnitude,
            yaw_offset_deg=math.degrees(math.atan2(nx, 1.0)),
            elevation_offset_deg=math.degrees(
                math.atan2(-ny, math.hypot(nx, 1.0))
            ),
        )


def _camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx_px, 0.0, intrinsics.cx_px],
            [0.0, intrinsics.fy_px, intrinsics.cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def calibrate_checkerboard_corners(
    views: Sequence[Sequence[Sequence[float]]],
    *,
    image_width: int,
    image_height: int,
    board_columns: int,
    board_rows: int,
    square_size: float,
    max_rms_px: float | None = None,
) -> CheckerboardCalibration:
    """Estimate lens parameters using multiple views of known inner corners."""

    if any(
        type(value) is not int or value < 3
        for value in (board_columns, board_rows)
    ):
        raise ValueError("Checkerboard dimensions must be integers >= 3")

    if any(
        type(value) is not int or value < 2
        for value in (image_width, image_height)
    ):
        raise ValueError("Image dimensions must be integers >= 2")

    if (
        isinstance(square_size, bool)
        or not isinstance(square_size, (int, float))
        or not math.isfinite(square_size)
        or square_size <= 0
    ):
        raise ValueError("Invalid checkerboard square size")

    if max_rms_px is not None and (
        isinstance(max_rms_px, bool)
        or not isinstance(max_rms_px, (int, float))
        or not math.isfinite(max_rms_px)
        or max_rms_px <= 0
    ):
        raise ValueError("Invalid RMS quality threshold")

    if len(views) < 6:
        raise ValueError("At least six checkerboard views required")

    grid = np.zeros(
        (board_rows * board_columns, 3),
        dtype=np.float32,
    )
    grid[:, :2] = np.mgrid[
        0:board_columns, 0:board_rows
    ].T.reshape(-1, 2)
    grid[:, :2] *= float(square_size)

    object_points = []
    image_points = []
    descriptors = []

    for index, view in enumerate(views):
        try:
            points = np.asarray(view, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"View {index}: invalid corner coordinates"
            ) from exc

        if points.shape != (board_rows * board_columns, 2):
            raise ValueError(
                f"View {index}: invalid corner count or shape"
            )

        if not bool(np.isfinite(points).all()):
            raise ValueError(f"View {index}: nonfinite coordinates")

        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] >= image_width)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] >= image_height)
        ):
            raise ValueError(f"View {index}: corners outside image")

        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        span = maximum - minimum

        if span[0] < 10 or span[1] < 10:
            raise ValueError(f"View {index}: checkerboard too small")

        descriptors.append(
            (
                float(points[:, 0].mean()),
                float(points[:, 1].mean()),
                float(span[0]),
            )
        )

        object_points.append(grid.copy())
        image_points.append(
            points.astype(np.float32).reshape(-1, 1, 2)
        )

    diversity = np.asarray(descriptors, dtype=np.float64)

    if max(
        np.std(diversity[:, 0]),
        np.std(diversity[:, 1]),
        np.std(diversity[:, 2]),
    ) < 2.0:
        raise ValueError(
            "Insufficient diversity among checkerboard views"
        )

    try:
        _, matrix, distortion, rotations, translations = (
            cv2.calibrateCamera(
                object_points,
                image_points,
                (image_width, image_height),
                None,
                None,
                flags=(
                    cv2.CALIB_FIX_K4
                    | cv2.CALIB_FIX_K5
                    | cv2.CALIB_FIX_K6
                ),
            )
        )
    except cv2.error as exc:
        raise ValueError("OpenCV calibration failed") from exc

    if (
        not bool(np.isfinite(matrix).all())
        or not bool(np.isfinite(distortion).all())
    ):
        raise ValueError("Calibration produced nonfinite parameters")

    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Invalid calibrated focal length")

    if not (
        0 <= matrix[0, 2] < image_width
        and 0 <= matrix[1, 2] < image_height
    ):
        raise ValueError("Calibrated principal point outside image")

    errors = []

    for obj, observed, rotation, translation in zip(
        object_points,
        image_points,
        rotations,
        translations,
    ):
        projected, _ = cv2.projectPoints(
            obj,
            rotation,
            translation,
            matrix,
            distortion,
        )

        difference = (
            projected.reshape(-1, 2).astype(np.float64)
            - observed.reshape(-1, 2)
        )

        error = float(
            np.sqrt(
                np.mean(np.sum(difference * difference, axis=1))
            )
        )
        errors.append(error)

    rms = float(np.sqrt(np.mean(np.square(errors))))

    if not math.isfinite(rms):
        raise ValueError("Invalid calibration RMS")

    if max_rms_px is not None and rms > max_rms_px:
        raise ValueError(
            f"RMS {rms:.3f}px exceeds threshold {max_rms_px:.3f}px"
        )

    coefficients = tuple(
        float(value)
        for value in distortion.ravel()[:5]
    )

    if len(coefficients) != 5:
        raise ValueError("Unexpected lens distortion parameters")

    return CheckerboardCalibration(
        intrinsics=CameraIntrinsics(
            width_px=image_width,
            height_px=image_height,
            fx_px=float(matrix[0, 0]),
            fy_px=float(matrix[1, 1]),
            cx_px=float(matrix[0, 2]),
            cy_px=float(matrix[1, 2]),
            provenance="measured",
        ),
        distortion=coefficients,
        rms_px=rms,
        per_view_rmse_px=tuple(errors),
        views=len(views),
        board_columns=board_columns,
        board_rows=board_rows,
    )
