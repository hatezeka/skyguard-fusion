"""Global association of predicted Kalman tracks and 2D detections."""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.detection.kalman_cv import KalmanCV2D
from app.detection.mahalanobis_gate import CHI2_2D_95


@dataclass(frozen=True)
class AssociationMatch:
    track_index: int
    detection_index: int
    distance_squared: float


@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[AssociationMatch, ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]
    best_cost: float
    alternative_cost: float | None
    cost_gap: float | None
    globally_ambiguous: bool


def associate_global(
    tracks: Sequence[KalmanCV2D],
    detections: Sequence[tuple[float, float]],
    gate_threshold: float = CHI2_2D_95,
    unmatched_cost: float | None = None,
    ambiguity_gap: float = 0.5,
) -> AssociationResult:
    """Associate predicted tracks without mutating Kalman filters.

    Call predict() on each track before invoking this function.
    Global ambiguity describes the assignment, not an individual ID.
    """

    if not isfinite(gate_threshold) or gate_threshold <= 0:
        raise ValueError("gate_threshold must be positive and finite")

    if not isfinite(ambiguity_gap) or ambiguity_gap < 0:
        raise ValueError("ambiguity_gap must be finite and nonnegative")

    if unmatched_cost is None:
        unmatched_cost = gate_threshold + 1.0

    if not isfinite(unmatched_cost) or unmatched_cost <= 0:
        raise ValueError("unmatched_cost must be positive and finite")

    n = len(tracks)
    m = len(detections)

    for x, y in detections:
        if not isfinite(x) or not isfinite(y):
            raise ValueError("detection coordinates must be finite")

    if n == 0:
        return AssociationResult(
            (), (), tuple(range(m)), 0.0, None, None, False
        )

    if m == 0:
        return AssociationResult(
            (), tuple(range(n)), (), n * unmatched_cost,
            None, None, False
        )

    distances = np.full((n, m), np.inf, dtype=float)

    for i, tracker in enumerate(tracks):
        for j, (x, y) in enumerate(detections):
            distance = tracker.innovation_distance_squared(x, y)
            if isfinite(distance) and 0 <= distance <= gate_threshold:
                distances[i, j] = distance

    # Each track has its own dummy column for an unmatched decision.
    # A forbidden cost must exceed every feasible full assignment.
    forbidden = (n + 1) * (
        max(gate_threshold, unmatched_cost) + 1.0
    )

    costs = np.full((n, m + n), forbidden, dtype=float)
    costs[:, :m] = np.where(
        np.isfinite(distances), distances, forbidden
    )

    for i in range(n):
        costs[i, m + i] = unmatched_cost

    def solve(matrix):
        rows, cols = linear_sum_assignment(matrix)
        total = float(matrix[rows, cols].sum())
        pairs = tuple(zip(rows.tolist(), cols.tolist()))
        return total, pairs

    best_cost, best_pairs = solve(costs)

    matches = tuple(
        AssociationMatch(i, j, float(distances[i, j]))
        for i, j in best_pairs
        if j < m and np.isfinite(distances[i, j])
    )

    matched_tracks = {match.track_index for match in matches}
    matched_detections = {
        match.detection_index for match in matches
    }

    # Find the best different assignment by excluding one selected
    # row/column pair at a time. Dummy alternatives representing
    # the same unmatched outcome are not meaningful competitors.
    alternative_cost = None

    for i, j in best_pairs:
        if j >= m:
            continue

        modified = costs.copy()
        modified[i, j] = forbidden

        candidate_cost, candidate_pairs = solve(modified)

        if any(modified[r, c] >= forbidden for r, c in candidate_pairs):
            continue

        if (
            alternative_cost is None
            or candidate_cost < alternative_cost
        ):
            alternative_cost = candidate_cost

    gap = (
        None if alternative_cost is None
        else max(0.0, alternative_cost - best_cost)
    )

    return AssociationResult(
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
