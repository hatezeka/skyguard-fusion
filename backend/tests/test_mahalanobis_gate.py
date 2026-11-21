import pytest

from app.detection.kalman_cv import KalmanCV2D
from app.detection.mahalanobis_gate import (
    CHI2_2D_95,
    CHI2_2D_99,
    mahalanobis_gate,
)


def test_exact_prediction_is_accepted():
    tracker = KalmanCV2D(50, 50)
    tracker.predict()

    result = mahalanobis_gate(tracker, 50, 50)

    assert result.accepted
    assert result.distance_squared == pytest.approx(0)


def test_near_measurement_is_accepted():
    tracker = KalmanCV2D(50, 50)
    tracker.predict()

    result = mahalanobis_gate(tracker, 51, 50)

    assert result.accepted


def test_far_measurement_is_rejected():
    tracker = KalmanCV2D(50, 50)
    tracker.predict()

    result = mahalanobis_gate(tracker, 500, 500)

    assert not result.accepted


def test_distance_increases_with_displacement():
    tracker = KalmanCV2D(50, 50)
    tracker.predict()

    near = mahalanobis_gate(tracker, 51, 50)
    far = mahalanobis_gate(tracker, 100, 50)

    assert near.distance_squared < far.distance_squared


def test_99_percent_gate_is_wider():
    tracker = KalmanCV2D(0, 0)
    tracker.predict()

    threshold_95 = mahalanobis_gate(
        tracker, 10, 0, CHI2_2D_95
    )

    threshold_99 = mahalanobis_gate(
        tracker, 10, 0, CHI2_2D_99
    )

    assert threshold_99.threshold > threshold_95.threshold
    assert (
        threshold_99.distance_squared
        == pytest.approx(threshold_95.distance_squared)
    )


def test_invalid_threshold_rejected():
    tracker = KalmanCV2D(0, 0)

    with pytest.raises(ValueError):
        mahalanobis_gate(tracker, 0, 0, threshold=0)

    with pytest.raises(ValueError):
        mahalanobis_gate(tracker, 0, 0, threshold=-1)


def test_gate_does_not_modify_filter():
    tracker = KalmanCV2D(10, 20)
    tracker.predict()

    before_state = tracker.state.copy()
    before_covariance = tracker.P.copy()

    mahalanobis_gate(tracker, 15, 22)

    assert (tracker.state == before_state).all()
    assert (tracker.P == before_covariance).all()


def test_measurement_uncertainty_affects_gate():
    low_uncertainty = KalmanCV2D(
        0, 0,
        initial_position_variance=1,
        initial_velocity_variance=1,
    )

    high_uncertainty = KalmanCV2D(
        0, 0,
        initial_position_variance=100,
        initial_velocity_variance=100,
    )

    low_uncertainty.predict()
    high_uncertainty.predict()

    low = mahalanobis_gate(low_uncertainty, 10, 0)
    high = mahalanobis_gate(high_uncertainty, 10, 0)

    assert high.distance_squared < low.distance_squared
