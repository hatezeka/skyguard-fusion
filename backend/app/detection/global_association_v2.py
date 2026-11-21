"""Probabilistic global association for predicted Kalman tracks.

The unmatched-detection hypothesis is a clutter-reference baseline.
A matched detection receives a log likelihood-ratio cost against
that baseline. Track misses receive a separate penalty.

All Kalman tracks must have been predicted before calling this
function. This module does not mutate them.
"""

from dataclasses import dataclass
from math import isfinite, log, pi
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.detection.kalman_cv import KalmanCV2D
from app.detection.mahalanobis_gate import CHI2_2D_95


@dataclass(frozen=True)
class ProbabilisticMatch:
    track_index: int
    detection_index: int
    mahalanobis_squared: float
    association_cost: float


@dataclass(frozen=True)
class ProbabilisticAssociationResult:
    matches: tuple[ProbabilisticMatch, ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]
    best_cost: float
    alternative_cost: float | None
    cost_gap: float | None
    globally_ambiguous: bool


def associate_probabilistic(
    tracks: Sequence[KalmanCV2D],
    detections: Sequence[tuple[float, float]],
    detection_probability: float = 0.90,
    clutter_density: float = 0.0001,
    gate_threshold: float = CHI2_2D_95,
    ambiguity_gap: float = 0.5,
) -> ProbabilisticAssociationResult:
    """Solve a gated, one-to-one likelihood-ratio association."""

    if not (0 < detection_probability < 1):
        raise ValueError("detection_probability must be in (0,1)")
    if not isfinite(clutter_density) or clutter_density <= 0:
        raise ValueError("clutter_density must be finite and positive")
    if not isfinite(gate_threshold) or gate_threshold <= 0:
        raise ValueError("gate_threshold must be finite and positive")
    if not isfinite(ambiguity_gap) or ambiguity_gap < 0:
        raise ValueError("ambiguity_gap must be finite and nonnegative")

    for x, y in detections:
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Detection coordinates must be finite")

    n = len(tracks)
    m = len(detections)

    miss_cost = -log(1.0 - detection_probability)

    if n == 0:
        return ProbabilisticAssociationResult(
            (), (), tuple(range(m)), 0.0, None, None, False
        )

    if m == 0:
        return ProbabilisticAssociationResult(
            (), tuple(range(n)), (),
            n * miss_cost, None, None, False
        )

    distances = np.full((n, m), np.inf)
    association_costs = np.full((n, m), np.inf)

    for i, track in enumerate(tracks):
        # KalmanCV2D stores state [x,y,vx,vy] and covariance P.
        # R is isotropic in the current Kalman implementation.
        predicted = track.current()
        mean = np.array([predicted.x, predicted.y], dtype=float)

        covariance = (
            track.P[:2, :2]
            + np.eye(2) * track.measurement_variance
        )

        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("Invalid innovation covariance")

        for j, (x, y) in enumerate(detections):
            d2 = track.innovation_distance_squared(x, y)

            if not isfinite(d2) or d2 < 0 or d2 > gate_threshold:
                continue

            log_density = -0.5 * (
                d2 + logdet + 2.0 * log(2.0 * pi)
            )

            cost = -(
                log(detection_probability)
                + log_density
                - log(clutter_density)
            )

            distances[i, j] = d2
            association_costs[i, j] = cost

    # Full assignment:
    # - n track rows
    # - m detection rows
    # - m detection columns
    # - n track-miss columns
    #
    # Detection rows select a zero-cost clutter baseline.
    # This permits negative log-likelihood-ratio match costs.

    size = n + m

    finite_costs = association_costs[
        np.isfinite(association_costs)
    ]
    maximum = (
        max(abs(float(v)) for v in finite_costs)
        if finite_costs.size else 0.0
    )
    forbidden = (size + 1) * (
        maximum + abs(miss_cost) + 1.0
    )

    matrix = np.full((size, size), forbidden, dtype=float)

    matrix[:n, :m] = np.where(
        np.isfinite(association_costs),
        association_costs,
        forbidden,
    )

    for i in range(n):
        matrix[i, m + i] = miss_cost

    # Each detection can be explained by the clutter baseline.
    for j in range(m):
        matrix[n + j, j] = 0.0

    # Dummy rows complete whichever miss columns remain.
    matrix[n:, m:] = 0.0

    def solve(cost_matrix):
        rows, cols = linear_sum_assignment(cost_matrix)
        pairs = tuple(zip(rows.tolist(), cols.tolist()))

        if any(cost_matrix[r, c] >= forbidden for r, c in pairs):
            return None

        total = float(cost_matrix[rows, cols].sum())
        return total, pairs

    solved = solve(matrix)
    if solved is None:
        raise RuntimeError("No feasible association")

    best_cost, best_pairs = solved

    matches = tuple(
        ProbabilisticMatch(
            i,
            j,
            float(distances[i, j]),
            float(association_costs[i, j]),
        )
        for i, j in best_pairs
        if i < n and j < m
    )

    matched_tracks = {match.track_index for match in matches}
    matched_detections = {
        match.detection_index for match in matches
    }

    # Enumerate the best alternative with a different set
    # of real track-to-detection matches.
    #
    # A distinct matching must either:
    # (a) remove a selected real edge, or
    # (b) introduce an unselected feasible real edge.
    #
    # Dummy-row permutations do not define distinct hypotheses.

    selected_edges = {
        (match.track_index, match.detection_index)
        for match in matches
    }

    alternative_cost = None

    def consider(candidate_matrix):
        nonlocal alternative_cost

        alternative = solve(candidate_matrix)
        if alternative is None:
            return

        candidate_cost, _ = alternative

        if (
            alternative_cost is None
            or candidate_cost < alternative_cost
        ):
            alternative_cost = candidate_cost

    # Remove each selected edge.
    for i, j in selected_edges:
        modified = matrix.copy()
        modified[i, j] = forbidden
        consider(modified)

    # Force each feasible edge that was not selected.
    for i in range(n):
        for j in range(m):
            if not np.isfinite(association_costs[i, j]):
                continue

            if (i, j) in selected_edges:
                continue

            modified = matrix.copy()

            # Force row i to use detection j.
            modified[i, :] = forbidden
            modified[i, j] = association_costs[i, j]

            # Prevent any other row from using detection j.
            modified[:, j] = forbidden
            modified[i, j] = association_costs[i, j]

            consider(modified)

    gap = (
        None if alternative_cost is None
        else max(0.0, alternative_cost - best_cost)
    )

    return ProbabilisticAssociationResult(
        matches=matches,
        unmatched_tracks=tuple(
            i for i in range(n) if i not in matched_tracks
        ),
        unmatched_detections=tuple(
            j for j in range(m) if j not in matched_detections
        ),
        best_cost=best_cost,
        alternative_cost=alternative_cost,
        cost_gap=gap,
        globally_ambiguous=(
            gap is not None and gap <= ambiguity_gap
        ),
    )
