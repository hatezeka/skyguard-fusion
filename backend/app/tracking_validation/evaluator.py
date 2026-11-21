from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True)
class IdentityMetrics:
    observations: int
    matched_observations: int
    id_switches: int
    identity_accuracy: float


def evaluate_identity_sequence(
    frames: list[dict[str, Hashable | None]],
) -> IdentityMetrics:
    previous_ids: dict[str, Hashable] = {}
    observations = 0
    matched = 0
    switches = 0

    for frame in frames:
        for identity, track_id in frame.items():
            observations += 1

            if track_id is None:
                continue

            matched += 1

            if (
                identity in previous_ids
                and previous_ids[identity] != track_id
            ):
                switches += 1

            previous_ids[identity] = track_id

    accuracy = matched / observations if observations else 0.0

    return IdentityMetrics(
        observations=observations,
        matched_observations=matched,
        id_switches=switches,
        identity_accuracy=accuracy,
    )
