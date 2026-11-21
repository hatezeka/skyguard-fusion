import pytest

from app.tracking_validation.scenarios import (
    crossing_scenario,
    parallel_scenario,
)
from app.tracking_validation.evaluator import (
    evaluate_identity_sequence,
)


def test_crossing_has_two_objects():
    frames = crossing_scenario()
    assert len(frames) == 21
    assert all(len(frame.objects) == 2 for frame in frames)


def test_crossing_preserves_ground_truth_identity():
    frames = crossing_scenario()
    assert all(
        {obj.identity for obj in frame.objects}
        == {"DRONE_A", "DRONE_B"}
        for frame in frames
    )


@pytest.mark.parametrize("count", [2, 5, 10])
def test_parallel_object_counts(count):
    frames = parallel_scenario(count)
    assert all(len(frame.objects) == count for frame in frames)


def test_evaluator_detects_id_switch():
    metrics = evaluate_identity_sequence([
        {"DRONE_A": 1, "DRONE_B": 2},
        {"DRONE_A": 1, "DRONE_B": 2},
        {"DRONE_A": 2, "DRONE_B": 1},
    ])

    assert metrics.id_switches == 2
    assert metrics.matched_observations == 6


def test_evaluator_no_switches():
    metrics = evaluate_identity_sequence([
        {"DRONE_A": 1},
        {"DRONE_A": 1},
        {"DRONE_A": 1},
    ])

    assert metrics.id_switches == 0
    assert metrics.identity_accuracy == 1.0


def test_missing_observation():
    metrics = evaluate_identity_sequence([
        {"DRONE_A": 1},
        {"DRONE_A": None},
        {"DRONE_A": 1},
    ])

    assert metrics.observations == 3
    assert metrics.matched_observations == 2
    assert metrics.id_switches == 0
    assert metrics.identity_accuracy == pytest.approx(2 / 3)
