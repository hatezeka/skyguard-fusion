"""Exact small-scale Top-K identity association hypotheses."""

from dataclasses import dataclass
from math import isfinite, log, pi
from typing import Sequence

import numpy as np

from app.detection.kalman_cv import KalmanCV2D
from app.detection.mahalanobis_gate import CHI2_2D_95


@dataclass(frozen=True)
class IdentityHypothesis:
    matches: tuple[tuple[int, int], ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]
    cost: float


def generate_hypotheses(
    tracks: Sequence[KalmanCV2D],
    detections: Sequence[tuple[float, float]],
    max_hypotheses: int = 8,
    detection_probability: float = 0.9,
    clutter_density: float = 0.0001,
    gate_threshold: float = CHI2_2D_95,
) -> tuple[IdentityHypothesis, ...]:
    """Enumerate unique physical assignments, sorted by cost.

    Reference implementation only. Does not mutate Kalman filters.
    All tracks must be predicted before this call.
    """

    if (
        isinstance(max_hypotheses, bool)
        or not isinstance(max_hypotheses, int)
        or max_hypotheses < 1
    ):
        raise ValueError("max_hypotheses must be positive integer")

    if not (
        isfinite(detection_probability)
        and 0 < detection_probability < 1
    ):
        raise ValueError("Invalid detection probability")

    if not isfinite(clutter_density) or clutter_density <= 0:
        raise ValueError("Invalid clutter density")

    if not isfinite(gate_threshold) or gate_threshold <= 0:
        raise ValueError("Invalid gate threshold")

    n = len(tracks)
    m = len(detections)

    if n > 4 or m > 6:
        raise ValueError(
            "Reference generator supports at most "
            "4 tracks and 6 detections"
        )

    for x, y in detections:
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Invalid detection coordinates")

    miss_cost = -log(1.0 - detection_probability)
    edge_costs = {}

    for i, track in enumerate(tracks):
        S = track.P[:2, :2] + track.R

        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0 or not isfinite(logdet):
            raise ValueError("Invalid innovation covariance")

        for j, (x, y) in enumerate(detections):
            d2 = track.innovation_distance_squared(x, y)

            if not isfinite(d2) or d2 < 0 or d2 > gate_threshold:
                continue

            log_density = -0.5 * (
                d2 + logdet + 2.0 * log(2.0 * pi)
            )

            edge_costs[i, j] = -(
                log(detection_probability)
                + log_density
                - log(clutter_density)
            )

    results = []

    def visit(track_index, used_detections, matches, cost):
        if track_index == n:
            matched_tracks = {i for i, _ in matches}

            results.append(
                IdentityHypothesis(
                    matches=tuple(matches),
                    unmatched_tracks=tuple(
                        i for i in range(n)
                        if i not in matched_tracks
                    ),
                    unmatched_detections=tuple(
                        j for j in range(m)
                        if j not in used_detections
                    ),
                    cost=float(cost),
                )
            )
            return

        visit(
            track_index + 1,
            used_detections,
            matches,
            cost + miss_cost,
        )

        for j in range(m):
            if j in used_detections:
                continue

            if (track_index, j) not in edge_costs:
                continue

            visit(
                track_index + 1,
                used_detections | {j},
                matches + [(track_index, j)],
                cost + edge_costs[track_index, j],
            )

    visit(0, set(), [], 0.0)

    results.sort(
        key=lambda h: (h.cost, h.matches)
    )

    return tuple(results[:max_hypotheses])
