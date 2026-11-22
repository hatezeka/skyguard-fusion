"""Synthetic geometry tests. The real profile is checked by the verification script."""
import json

import numpy as np
import pytest

from app.localization.stereo_calibration import StereoCalibration


@pytest.fixture
def profile(tmp_path):
    k = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
    p1 = [row + [0] for row in k]
    p2 = [row[:] for row in p1]
    p2[0][3] = -145000
    rotation = np.eye(3).tolist()
    payload = {
        "resolution_px": [640, 480], "physical_baseline_mm": 290,
        "holdout_epipolar_mean_px": 0.2, "K1": k, "K2": k,
        "D1": [0] * 5, "D2": [0] * 5,
        "R": rotation, "R1": rotation, "R2": rotation,
        "T_mm_nominal": [-290, 0, 0], "P1": p1, "P2": p2, "Q": rotation,
    }
    payload["Q"] = [[1, 0, 0, -320], [0, 1, 0, -240],
                    [0, 0, 0, 500], [0, 0, 1 / 290, 0]]
    file = tmp_path / "profile.json"
    file.write_text(json.dumps(payload), encoding="utf-8")
    return file


def test_synthetic_matching_points_reconstruct_metric_depth(profile):
    model = StereoCalibration.load(profile)
    vector = model.triangulate_rectified((320, 240), (175, 240), max_epipolar_px=1)
    assert vector == pytest.approx([0, 0, 1], abs=1e-9)


@pytest.mark.parametrize(("right", "error"), [
    ((175, 250), "Epipolar"),
    ((320, 240), "disparity"),
    ((400, 240), "depth"),
    ((float("nan"), 240), "Invalid image point"),
])
def test_inconsistent_observation_rejected(profile, right, error):
    with pytest.raises(ValueError, match=error):
        StereoCalibration.load(profile).triangulate_rectified((320, 240), right, max_epipolar_px=1)


def test_invalid_matrix_shape_rejected(profile):
    payload = json.loads(profile.read_text())
    payload["P2"] = [[0]]
    profile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid shape for P2"):
        StereoCalibration.load(profile)
