"""Constant-velocity Kalman filter for 2D object tracking."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanPrediction:
    x: float
    y: float
    vx: float
    vy: float
    position_covariance: np.ndarray


class KalmanCV2D:
    """2D Kalman filter with state [x, y, vx, vy]."""

    def __init__(
        self,
        x: float,
        y: float,
        process_variance: float = 4.0,
        measurement_variance: float = 4.0,
        initial_position_variance: float = 4.0,
        initial_velocity_variance: float = 100.0,
    ):
        values = (
            x,
            y,
            process_variance,
            measurement_variance,
            initial_position_variance,
            initial_velocity_variance,
        )

        if not all(np.isfinite(value) for value in values):
            raise ValueError("All parameters must be finite")

        if (
            process_variance <= 0
            or measurement_variance <= 0
            or initial_position_variance <= 0
            or initial_velocity_variance <= 0
        ):
            raise ValueError("Variances must be positive")

        self.state = np.array(
            [x, y, 0.0, 0.0], dtype=np.float64
        )

        self.P = np.diag([
            initial_position_variance,
            initial_position_variance,
            initial_velocity_variance,
            initial_velocity_variance,
        ]).astype(np.float64)

        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        self.R = np.eye(2) * self.measurement_variance

    def predict(self, dt: float = 1.0) -> KalmanPrediction:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        G = np.array([
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ])

        Q = self.process_variance * (G @ G.T)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        self.P = (self.P + self.P.T) / 2.0

        return self.current()

    def update(self, x: float, y: float) -> KalmanPrediction:
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Measurements must be finite")

        z = np.array([x, y], dtype=np.float64)

        innovation = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R

        K = np.linalg.solve(
            S,
            self.H @ self.P,
        ).T

        self.state = self.state + K @ innovation

        I = np.eye(4)
        A = I - K @ self.H

        # Joseph form improves numerical stability.
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2.0

        return self.current()

    def innovation_distance_squared(
        self, x: float, y: float
    ) -> float:
        """Squared Mahalanobis distance in measurement space."""
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Measurements must be finite")

        z = np.array([x, y], dtype=np.float64)
        residual = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R

        return float(residual @ np.linalg.solve(S, residual))

    def current(self) -> KalmanPrediction:
        return KalmanPrediction(
            x=float(self.state[0]),
            y=float(self.state[1]),
            vx=float(self.state[2]),
            vy=float(self.state[3]),
            position_covariance=self.P[:2, :2].copy(),
        )
