import numpy as np
import pytest

from app.detection.kalman_cv import KalmanCV2D


def test_initial_state():
    kf = KalmanCV2D(10, 20)
    state = kf.current()

    assert state.x == 10
    assert state.y == 20
    assert state.vx == 0
    assert state.vy == 0


def test_prediction_increases_uncertainty():
    kf = KalmanCV2D(10, 20)

    before = np.trace(kf.current().position_covariance)
    after = np.trace(kf.predict().position_covariance)

    assert after > before


def test_update_moves_toward_measurement():
    kf = KalmanCV2D(0, 0)
    kf.predict()

    before = abs(kf.current().x - 10)
    after = abs(kf.update(10, 0).x - 10)

    assert after < before


def test_constant_velocity_tracking():
    kf = KalmanCV2D(0, 0)

    for frame in range(1, 21):
        kf.predict()
        state = kf.update(5 * frame, 2 * frame)

    assert abs(state.x - 100) < 2
    assert abs(state.y - 40) < 2
    assert abs(state.vx - 5) < 1
    assert abs(state.vy - 2) < 1


def test_mahalanobis_near_vs_far():
    kf = KalmanCV2D(50, 50)
    kf.predict()

    near = kf.innovation_distance_squared(51, 50)
    far = kf.innovation_distance_squared(100, 100)

    assert near < far


def test_invalid_dt_rejected():
    kf = KalmanCV2D(0, 0)

    with pytest.raises(ValueError):
        kf.predict(0)

    with pytest.raises(ValueError):
        kf.predict(-1)


def test_invalid_variances_rejected():
    with pytest.raises(ValueError):
        KalmanCV2D(0, 0, measurement_variance=0)


def test_covariance_remains_symmetric_and_positive():
    kf = KalmanCV2D(0, 0)

    for frame in range(1, 101):
        kf.predict()
        kf.update(frame * 2.0, frame * 0.5)

    assert np.allclose(kf.P, kf.P.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(kf.P)) > 0
