"""Calibrate camera intrinsics from checkerboard photographs.

Only camera-space geometry is estimated. RMS is a reprojection
metric, not an independent physical-accuracy guarantee.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.localization.checkerboard_calibration import (
    CheckerboardCalibration,
    calibrate_checkerboard_corners,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_IMAGES = 64
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 24_000_000


@dataclass(frozen=True)
class ImageAssessment:
    filename: str
    accepted: bool
    reason: str
    sharpness: float | None
    coverage_fraction: float | None


@dataclass(frozen=True)
class ImageCalibrationReport:
    calibration: CheckerboardCalibration
    image_width: int
    image_height: int
    assessments: tuple[ImageAssessment, ...]

    def as_dict(self) -> dict:
        return {
            "model": "pinhole_radial_tangential_5",
            "validation": "reprojection_only",
            "physical_range_estimated": False,
            "object_gps_estimated": False,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "intrinsics": asdict(self.calibration.intrinsics),
            "distortion": list(self.calibration.distortion),
            "rms_px": self.calibration.rms_px,
            "per_view_rmse_px": list(self.calibration.per_view_rmse_px),
            "accepted_views": self.calibration.views,
            "assessments": [asdict(item) for item in self.assessments],
        }


def _finite_number(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def calibrate_from_images(
    image_paths: Sequence[str | Path],
    *,
    board_columns: int,
    board_rows: int,
    square_size_m: float,
    min_sharpness: float = 20.0,
    min_coverage_fraction: float = 0.015,
    max_rms_px: float = 1.0,
) -> ImageCalibrationReport:
    """Detect inner checkerboard corners in six or more diverse views.

    All accepted images must have identical resolution. Lens settings,
    focus and zoom should remain unchanged during image acquisition.
    """

    if isinstance(image_paths, (str, Path)):
        raise ValueError("Provide a sequence of image paths")

    paths = [Path(path) for path in image_paths]

    if not 6 <= len(paths) <= MAX_IMAGES:
        raise ValueError("Provide 6 to 64 image paths")

    if len({str(path.resolve()) for path in paths}) != len(paths):
        raise ValueError("Duplicate image paths are not permitted")

    if (
        type(board_columns) is not int
        or type(board_rows) is not int
        or board_columns < 3
        or board_rows < 3
    ):
        raise ValueError("Invalid checkerboard inner-corner dimensions")

    if _finite_number(square_size_m, "square_size_m") <= 0:
        raise ValueError("square_size_m must be positive")

    sharpness_threshold = _finite_number(
        min_sharpness, "min_sharpness"
    )
    coverage_threshold = _finite_number(
        min_coverage_fraction, "min_coverage_fraction"
    )
    rms_threshold = _finite_number(max_rms_px, "max_rms_px")

    if sharpness_threshold < 0:
        raise ValueError("min_sharpness must be nonnegative")
    if not 0 <= coverage_threshold < 1:
        raise ValueError("Invalid coverage threshold")
    if rms_threshold <= 0:
        raise ValueError("max_rms_px must be positive")

    assessments: list[ImageAssessment] = []
    observations: list[list[list[float]]] = []
    expected_size: tuple[int, int] | None = None

    pattern_size = (board_columns, board_rows)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        35,
        0.001,
    )

    for path in paths:
        reason = None
        sharpness = None
        coverage = None
        filename = path.name

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            reason = "unsupported_extension"
        elif not path.is_file():
            reason = "file_not_found"
        elif path.stat().st_size > MAX_FILE_BYTES:
            reason = "file_too_large"

        if reason is not None:
            assessments.append(
                ImageAssessment(filename, False, reason, None, None)
            )
            continue

        try:
            encoded = np.fromfile(str(path), dtype=np.uint8)
            gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        except (OSError, ValueError, cv2.error):
            gray = None

        if gray is None:
            reason = "unreadable_image"
        else:
            height, width = gray.shape

            if width * height > MAX_PIXELS:
                reason = "image_dimensions_too_large"
            elif (
                expected_size is not None
                and expected_size != (width, height)
            ):
                reason = "resolution_mismatch"
            else:
                sharpness = float(
                    cv2.Laplacian(gray, cv2.CV_64F).var()
                )

                if sharpness < sharpness_threshold:
                    reason = "insufficient_sharpness"
                else:
                    found, corners = cv2.findChessboardCorners(
                        gray, pattern_size, flags
                    )

                    if not found or corners is None:
                        reason = "corners_not_found"
                    else:
                        corners = cv2.cornerSubPix(
                            gray, corners, (7, 7), (-1, -1), criteria
                        )
                        points = corners.reshape(-1, 2)
                        span = np.ptp(points, axis=0)

                        coverage = float(
                            span[0] * span[1] / (width * height)
                        )

                        if coverage < coverage_threshold:
                            reason = "insufficient_board_coverage"
                        else:
                            if expected_size is None:
                                expected_size = (width, height)
                            observations.append(
                                points.astype(float).tolist()
                            )

        assessments.append(
            ImageAssessment(
                filename=filename,
                accepted=reason is None,
                reason=reason or "accepted",
                sharpness=sharpness,
                coverage_fraction=coverage,
            )
        )

    if len(observations) < 6:
        failures = Counter(
            item.reason for item in assessments
            if not item.accepted
        )
        raise ValueError(
            f"Only {len(observations)} valid checkerboard views; "
            f"at least 6 required. Rejections: {dict(failures)}"
        )

    assert expected_size is not None
    width, height = expected_size

    calibration = calibrate_checkerboard_corners(
        observations,
        image_width=width,
        image_height=height,
        board_columns=board_columns,
        board_rows=board_rows,
        square_size=square_size_m,
        max_rms_px=rms_threshold,
    )

    return ImageCalibrationReport(
        calibration=calibration,
        image_width=width,
        image_height=height,
        assessments=tuple(assessments),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkyGuard checkerboard camera calibration"
    )
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--max-rms-px", type=float, default=1.0)
    parser.add_argument("--min-sharpness", type=float, default=20.0)
    args = parser.parse_args()

    folder = args.image_directory

    if not folder.is_dir():
        parser.error("Image directory does not exist")

    paths = sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    try:
        result = calibrate_from_images(
            paths,
            board_columns=args.columns,
            board_rows=args.rows,
            square_size_m=args.square_size_m,
            min_sharpness=args.min_sharpness,
            max_rms_px=args.max_rms_px,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
