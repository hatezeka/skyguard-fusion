"""Out-of-fold geometric validation of calibrated camera intrinsics.

A held-out image is excluded from intrinsic fitting. Its pose is fitted
using half of its checkerboard corners, and reprojection is evaluated on
the other half. This is NOT independent physical range/GPS validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from app.localization.checkerboard_calibration import calibrate_checkerboard_corners


@dataclass(frozen=True)
class CalibrationFold:
    fold: int
    training_views: int
    heldout_views: int
    training_rms_px: float
    heldout_rmse_px: float
    heldout_per_view_rmse_px: tuple[float, ...]
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float


@dataclass(frozen=True)
class CalibrationValidation:
    total_views: int
    folds: tuple[CalibrationFold, ...]
    heldout_rmse_px: float
    worst_heldout_view_rmse_px: float
    fx_spread_pct: float
    fy_spread_pct: float
    principal_point_max_shift_px: float
    validation_type: str = "heldout_images_and_corners_reprojection"
    physical_range_validated: bool = False
    object_gps_validated: bool = False


def _positive_limit(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def validate_calibration_stability(
    views: Sequence[Sequence[Sequence[float]]],
    *,
    image_width: int,
    image_height: int,
    board_columns: int,
    board_rows: int,
    square_size_m: float,
    max_training_rms_px: float | None = None,
    max_heldout_view_rmse_px: float | None = None,
    max_focal_spread_pct: float | None = None,
    max_principal_shift_px: float | None = None,
) -> CalibrationValidation:
    """Four deterministic leave-one-quarter-out calibration folds.

    Each held-out checkerboard pose is fitted using alternating corners;
    the remaining corners are used for the reprojection score.
    """
    if isinstance(views, (str, bytes)) or not 10 <= len(views) <= 64:
        raise ValueError("Provide 10 to 64 checkerboard views")
    if any(type(v) is not int or v < 2 or v > 10000 for v in (image_width, image_height)):
        raise ValueError("Invalid image dimensions")
    if any(type(v) is not int or not 3 <= v <= 30 for v in (board_columns, board_rows)):
        raise ValueError("Invalid checkerboard dimensions")
    if isinstance(square_size_m, bool) or not isinstance(square_size_m, (int, float)) or not math.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError("square_size_m must be finite and positive")
    training_limit = _positive_limit(max_training_rms_px, "max_training_rms_px")
    holdout_limit = _positive_limit(max_heldout_view_rmse_px, "max_heldout_view_rmse_px")
    focal_limit = _positive_limit(max_focal_spread_pct, "max_focal_spread_pct")
    principal_limit = _positive_limit(max_principal_shift_px, "max_principal_shift_px")

    observed = []
    for i, view in enumerate(views):
        try:
            arr = np.asarray(view, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"View {i}: invalid coordinates") from exc
        if arr.shape != (board_columns * board_rows, 2) or not np.isfinite(arr).all():
            raise ValueError(f"View {i}: invalid or nonfinite corners")
        if np.any(arr[:, 0] < 0) or np.any(arr[:, 0] >= image_width) or np.any(arr[:, 1] < 0) or np.any(arr[:, 1] >= image_height):
            raise ValueError(f"View {i}: corner outside image")
        observed.append(arr)

    object_points = np.zeros((board_rows * board_columns, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:board_columns, 0:board_rows].T.reshape(-1, 2)
    object_points[:, :2] *= float(square_size_m)
    row, col = np.indices((board_rows, board_columns))
    fitting_mask = ((row + col).reshape(-1) % 2 == 0)
    checking_mask = ~fitting_mask

    scores: list[CalibrationFold] = []
    all_rmse: list[float] = []
    for fold in range(4):
        heldout = [i for i in range(len(observed)) if i % 4 == fold]
        training = [i for i in range(len(observed)) if i % 4 != fold]
        try:
            model = calibrate_checkerboard_corners(
                [observed[i].tolist() for i in training],
                image_width=image_width,
                image_height=image_height,
                board_columns=board_columns,
                board_rows=board_rows,
                square_size=float(square_size_m),
                max_rms_px=training_limit,
            )
        except (ValueError, cv2.error) as exc:
            raise ValueError(f"Fold {fold}: calibration failed: {exc}") from exc

        intr = model.intrinsics
        matrix = np.array([
            [intr.fx_px, 0, intr.cx_px],
            [0, intr.fy_px, intr.cy_px],
            [0, 0, 1],
        ], dtype=np.float64)
        distortion = np.asarray(model.distortion, dtype=np.float64)
        errors = []
        for index in heldout:
            points = observed[index]
            try:
                ok, rotation, translation = cv2.solvePnP(
                    object_points[fitting_mask],
                    points[fitting_mask].reshape(-1, 1, 2),
                    matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    raise ValueError("pose estimation did not converge")
                predicted, _ = cv2.projectPoints(
                    object_points[checking_mask], rotation, translation,
                    matrix, distortion,
                )
                differences = predicted.reshape(-1, 2) - points[checking_mask]
                rmse = float(np.sqrt(np.mean(np.sum(differences**2, axis=1))))
            except (ValueError, cv2.error) as exc:
                raise ValueError(f"Fold {fold}, view {index}: invalid heldout pose: {exc}") from exc
            if not math.isfinite(rmse):
                raise ValueError(f"Fold {fold}, view {index}: nonfinite error")
            errors.append(rmse)
            all_rmse.append(rmse)

        scores.append(CalibrationFold(
            fold=fold,
            training_views=len(training),
            heldout_views=len(heldout),
            training_rms_px=model.rms_px,
            heldout_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),
            heldout_per_view_rmse_px=tuple(errors),
            fx_px=intr.fx_px, fy_px=intr.fy_px,
            cx_px=intr.cx_px, cy_px=intr.cy_px,
        ))

    focal_x = [score.fx_px for score in scores]
    focal_y = [score.fy_px for score in scores]
    fx_spread = 100 * (max(focal_x) - min(focal_x)) / (sum(focal_x) / len(focal_x))
    fy_spread = 100 * (max(focal_y) - min(focal_y)) / (sum(focal_y) / len(focal_y))
    shift = max(
        math.hypot(a.cx_px - b.cx_px, a.cy_px - b.cy_px)
        for a in scores for b in scores
    )
    worst = max(all_rmse)
    if holdout_limit is not None and worst > holdout_limit:
        raise ValueError(f"Worst heldout RMSE {worst:.4f}px exceeds {holdout_limit:.4f}px")
    if focal_limit is not None and max(fx_spread, fy_spread) > focal_limit:
        raise ValueError("Focal-length instability exceeds threshold")
    if principal_limit is not None and shift > principal_limit:
        raise ValueError("Principal-point instability exceeds threshold")

    return CalibrationValidation(
        total_views=len(observed), folds=tuple(scores),
        heldout_rmse_px=float(np.sqrt(np.mean(np.square(all_rmse)))),
        worst_heldout_view_rmse_px=worst,
        fx_spread_pct=fx_spread, fy_spread_pct=fy_spread,
        principal_point_max_shift_px=shift,
    )
