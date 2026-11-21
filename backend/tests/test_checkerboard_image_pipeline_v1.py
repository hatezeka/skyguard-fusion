"""End-to-end image acquisition tests using rendered synthetic boards."""

import cv2
import numpy as np
import pytest

from app.localization.checkerboard_image_pipeline import (
    calibrate_from_images,
)

COLS, ROWS = 7, 5
WIDTH, HEIGHT = 1280, 720
SQUARE = 0.05

K = np.array([
    [920.0, 0.0, 640.0],
    [0.0, 910.0, 360.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def make_images(folder, count=8):
    square_pixels = 100
    board = np.full(
        ((ROWS + 1) * square_pixels,
         (COLS + 1) * square_pixels),
        255,
        dtype=np.uint8,
    )

    for row in range(ROWS + 1):
        for column in range(COLS + 1):
            if (row + column) % 2 == 0:
                y = row * square_pixels
                x = column * square_pixels
                board[
                    y:y + square_pixels,
                    x:x + square_pixels,
                ] = 0

    object_outer = np.array([
        [-SQUARE, -SQUARE, 0],
        [COLS * SQUARE, -SQUARE, 0],
        [COLS * SQUARE, ROWS * SQUARE, 0],
        [-SQUARE, ROWS * SQUARE, 0],
    ], dtype=np.float32)

    source_corners = np.float32([
        [0, 0],
        [board.shape[1] - 1, 0],
        [board.shape[1] - 1, board.shape[0] - 1],
        [0, board.shape[0] - 1],
    ])

    files = []

    for index in range(count):
        rotation = np.array([
            -0.18 + 0.06 * (index % 4),
            -0.23 + 0.07 * (index % 5),
            -0.10 + 0.06 * (index % 3),
        ])
        translation = np.array([
            -0.19 + 0.08 * (index % 5),
            -0.12 + 0.055 * (index % 4),
            0.80 + 0.08 * (index % 4),
        ])

        projected, _ = cv2.projectPoints(
            object_outer,
            rotation,
            translation,
            K,
            np.zeros(5),
        )

        transform = cv2.getPerspectiveTransform(
            source_corners,
            projected.reshape(-1, 2).astype(np.float32),
        )

        canvas = np.full(
            (HEIGHT, WIDTH), 185, dtype=np.uint8
        )

        warped = cv2.warpPerspective(
            board, transform, (WIDTH, HEIGHT),
            borderValue=185,
        )
        mask = cv2.warpPerspective(
            np.full_like(board, 255),
            transform,
            (WIDTH, HEIGHT),
            borderValue=0,
        )

        canvas[mask > 0] = warped[mask > 0]

        path = folder / f"board_{index:02d}.png"
        assert cv2.imwrite(str(path), canvas)
        files.append(path)

    return files


def run_pipeline(paths, **kwargs):
    return calibrate_from_images(
        paths,
        board_columns=COLS,
        board_rows=ROWS,
        square_size_m=SQUARE,
        **kwargs,
    )


def test_real_image_reader_and_corner_detector(tmp_path):
    files = make_images(tmp_path)
    result = run_pipeline(files)

    assert result.calibration.views == 8
    assert result.calibration.rms_px < 1.0
    assert len(result.calibration.per_view_rmse_px) == 8
    assert all(item.accepted for item in result.assessments)

    assert result.calibration.intrinsics.fx_px == pytest.approx(
        920, abs=100
    )
    assert result.calibration.intrinsics.fy_px == pytest.approx(
        910, abs=100
    )

    report = result.as_dict()
    assert report["physical_range_estimated"] is False
    assert report["object_gps_estimated"] is False
    assert report["validation"] == "reprojection_only"


def test_rejected_images_are_reported(tmp_path):
    files = make_images(tmp_path)

    blank = tmp_path / "blank.png"
    wrong_size = tmp_path / "wrong_size.png"

    assert cv2.imwrite(
        str(blank),
        np.full((HEIGHT, WIDTH), 185, dtype=np.uint8),
    )
    assert cv2.imwrite(
        str(wrong_size),
        np.full((400, 500), 185, dtype=np.uint8),
    )

    result = run_pipeline(files + [blank, wrong_size])
    reasons = {item.reason for item in result.assessments}

    assert result.calibration.views == 8
    assert "insufficient_sharpness" in reasons
    assert "resolution_mismatch" in reasons


def test_insufficient_input_images_fails(tmp_path):
    files = make_images(tmp_path, count=5)

    with pytest.raises(ValueError, match="Provide 6 to 64 image paths"):
        run_pipeline(files)


def test_insufficient_accepted_images_fails(tmp_path):
    files = make_images(tmp_path, count=8)

    # Eight submitted images, but only five contain a valid board.
    for path in files[5:]:
        assert cv2.imwrite(
            str(path),
            np.full((HEIGHT, WIDTH), 185, dtype=np.uint8),
        )

    with pytest.raises(
        ValueError,
        match="Only 5 valid checkerboard views; at least 6 required",
    ):
        run_pipeline(files)


def test_duplicate_files_are_rejected(tmp_path):
    files = make_images(tmp_path)

    with pytest.raises(ValueError, match="Duplicate"):
        run_pipeline(files + [files[0]])


def test_invalid_quality_threshold_rejected(tmp_path):
    files = make_images(tmp_path)

    with pytest.raises(ValueError):
        run_pipeline(files, min_sharpness=float("nan"))


def test_strict_rms_threshold_rejects_calibration(tmp_path):
    files = make_images(tmp_path)

    with pytest.raises(ValueError, match="RMS"):
        run_pipeline(files, max_rms_px=1e-12)
