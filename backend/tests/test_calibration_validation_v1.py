import cv2
import numpy as np
import pytest
from app.localization.calibration_validation import validate_calibration_stability

COLS, ROWS = 8, 6
W, H = 1280, 720
SQUARE = 0.035
K = np.array([[940.0, 0, 630.0],[0, 925.0, 355.0],[0, 0, 1.0]], dtype=np.float64)
DIST = np.array([0.016, -0.006, 0.001, -0.0008, 0.001])

@pytest.fixture(scope='module')
def views():
    board = np.zeros((ROWS * COLS, 3), dtype=np.float32)
    board[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE
    rng = np.random.default_rng(2026)
    data=[]
    for idx in range(15):
        r=np.array([0.04+(idx%4)*.075, -.18+(idx%5)*.083, -.12+(idx%3)*.06])
        t=np.array([-.20+(idx%5)*.075, -.13+(idx%4)*.035, .8+(idx%4)*.095])
        points,_=cv2.projectPoints(board,r,t,K,DIST)
        data.append((points.reshape(-1,2)+rng.normal(0,0.12,(len(board),2))).tolist())
    return data

def validate(views, **kw):
    return validate_calibration_stability(views, image_width=W, image_height=H,
        board_columns=COLS, board_rows=ROWS, square_size_m=SQUARE, **kw)

def test_four_folds_and_heldout_corners(views):
    r=validate(views,max_training_rms_px=0.5,max_heldout_view_rmse_px=0.8,
        max_focal_spread_pct=8,max_principal_shift_px=30)
    assert len(r.folds)==4
    assert sum(x.heldout_views for x in r.folds)==len(views)
    assert r.heldout_rmse_px < .5
    assert r.worst_heldout_view_rmse_px < .8
    assert not r.physical_range_validated and not r.object_gps_validated

def test_stable_parameters_near_ground_truth(views):
    r=validate(views)
    assert all(abs(f.fx_px-940)<70 for f in r.folds)
    assert all(abs(f.fy_px-925)<70 for f in r.folds)

def test_deterministic(views):
    assert validate(views)==validate(views)

def test_insufficient_views_rejected(views):
    with pytest.raises(ValueError,match='10 to 64'):
        validate(views[:9])

def test_nonfinite_corners_rejected(views):
    broken=np.asarray(views).copy()
    broken[0,0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        validate(broken)

def test_outside_image_rejected(views):
    broken=np.asarray(views).copy()
    broken[0,0,1]=-3
    with pytest.raises(ValueError,match='outside'):
        validate(broken)

def test_strict_quality_gate_rejected(views):
    with pytest.raises(ValueError, match='Worst heldout RMSE'):
        validate(views,max_heldout_view_rmse_px=1e-8)

def test_invalid_limit_rejected(views):
    with pytest.raises(ValueError,match='finite positive'):
        validate(views,max_focal_spread_pct=float('nan'))
