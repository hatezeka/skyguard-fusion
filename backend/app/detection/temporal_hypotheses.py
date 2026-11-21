"""Bounded temporal association hypotheses: fixed-track prototype."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

from app.detection.kalman_cv import KalmanCV2D
from app.detection.identity_hypotheses import generate_hypotheses


@dataclass
class TemporalBranch:
    filters: list[KalmanCV2D]
    cost: float
    history: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True)
class TemporalDecision:
    frame_index: int
    assignments: tuple[tuple[int, int], ...]
    ambiguous_tracks: tuple[int, ...]
    best_cost: float
    retained_hypotheses: int
    truncated: bool


class TemporalHypothesisManager:
    """Research prototype; track identities are fixed at initialization.

    Inputs are consecutive frames of detection centers.
    Track births, deaths and production confirmation are not implemented.
    """

    def __init__(
        self,
        initial_positions,
        max_hypotheses=8,
        ambiguity_gap=0.5,
        detection_probability=0.9,
        clutter_density=0.0001,
    ):
        if (
            isinstance(max_hypotheses, bool)
            or not isinstance(max_hypotheses, int)
            or not 1 <= max_hypotheses <= 8
        ):
            raise ValueError("max_hypotheses must be in [1,8]")

        if not isfinite(ambiguity_gap) or ambiguity_gap < 0:
            raise ValueError("Invalid ambiguity gap")

        if len(initial_positions) > 4:
            raise ValueError("At most 4 initial tracks")

        self.max_hypotheses = max_hypotheses
        self.ambiguity_gap = float(ambiguity_gap)
        self.detection_probability = detection_probability
        self.clutter_density = clutter_density

        self.branches = [
            TemporalBranch(
                filters=[KalmanCV2D(x, y) for x, y in initial_positions],
                cost=0.0,
                history=(),
            )
        ]

        self.last_frame = None

    def update(self, frame_index, detections):
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
            or (
                self.last_frame is not None
                and frame_index != self.last_frame + 1
            )
        ):
            raise ValueError("Frames must be consecutive")

        if len(detections) > 6:
            raise ValueError("At most 6 detections")

        for x, y in detections:
            if not isfinite(x) or not isfinite(y):
                raise ValueError("Invalid detection")

        # Work on copies: failed updates must not alter live branches.
        candidates = []

        for branch in self.branches:
            predicted = deepcopy(branch.filters)

            for track in predicted:
                track.predict()

            hypotheses = generate_hypotheses(
                predicted,
                detections,
                max_hypotheses=self.max_hypotheses,
                detection_probability=self.detection_probability,
                clutter_density=self.clutter_density,
            )

            for hypothesis in hypotheses:
                filters = deepcopy(predicted)

                for i, j in hypothesis.matches:
                    filters[i].update(*detections[j])

                candidates.append(
                    TemporalBranch(
                        filters=filters,
                        cost=branch.cost + hypothesis.cost,
                        history=branch.history + (hypothesis.matches,),
                    )
                )

        candidates.sort(key=lambda b: (b.cost, b.history))
        truncated = len(candidates) > self.max_hypotheses
        retained = candidates[:self.max_hypotheses]
        pruning_gap = (
            candidates[self.max_hypotheses].cost - retained[0].cost
            if truncated else None
        )
        pruning_uncertain = (
            pruning_gap is not None
            and pruning_gap <= self.ambiguity_gap + 1e-9
        )

        if not retained:
            raise RuntimeError("No temporal hypotheses")

        best = retained[0]

        # Compare histories sufficiently close in cumulative cost.
        close = [
            branch for branch in retained
            if branch.cost - best.cost <= self.ambiguity_gap + 1e-9
        ]

        # Conservative output: when pruning occurs, this prototype
        # does not certify that excluded histories are irrelevant.
        assignments = []
        ambiguous = []

        for track_index in range(len(best.filters)):
            choices = []

            for branch in close:
                mapping = dict(branch.history[-1])
                choices.append(mapping.get(track_index))

            if pruning_uncertain:
                ambiguous.append(track_index)
            elif len(set(choices)) == 1 and choices[0] is not None:
                assignments.append((track_index, choices[0]))
            else:
                ambiguous.append(track_index)

        self.branches = retained
        self.last_frame = frame_index

        return TemporalDecision(
            frame_index=frame_index,
            assignments=tuple(assignments),
            ambiguous_tracks=tuple(ambiguous),
            best_cost=best.cost,
            retained_hypotheses=len(retained),
            truncated=truncated,
        )
