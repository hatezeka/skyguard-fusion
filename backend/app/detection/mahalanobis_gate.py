"""Statistical measurement gating for 2D Kalman tracks."""

from dataclasses import dataclass
from math import isfinite

from app.detection.kalman_cv import KalmanCV2D


CHI2_2D_95 = 5.991464547107979
CHI2_2D_99 = 9.21034037197618


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    distance_squared: float
    threshold: float


def mahalanobis_gate(
    tracker: KalmanCV2D,
    x: float,
    y: float,
    threshold: float = CHI2_2D_95,
) -> GateResult:
    """Evaluate one measurement against the predicted track state.

    Call after predict() and before update().
    """

    if not isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")

    distance = tracker.innovation_distance_squared(x, y)

    if not isfinite(distance) or distance < 0:
        raise ValueError("invalid Mahalanobis distance")

    return GateResult(
        accepted=distance <= threshold,
        distance_squared=distance,
        threshold=threshold,
    )
