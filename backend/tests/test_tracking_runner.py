import pytest

from app.detection.tracker import ObjectTracker
from app.tracking_validation.runner import run_scenario
from app.tracking_validation.scenarios import (
    crossing_scenario,
    parallel_scenario,
)


@pytest.mark.parametrize("count", [2, 5, 10])
def test_parallel_tracking(count):
    result = run_scenario(
        f"parallel_{count}",
        parallel_scenario(count),
    )

    assert result.objects == count
    assert result.metrics.observations == count * 20
    assert result.metrics.matched_observations == count * 20
    assert result.metrics.id_switches == 0
    assert result.unique_track_ids == count


def test_crossing_tracking_metrics():
    result = run_scenario(
        "crossing_2",
        crossing_scenario(),
    )

    assert result.objects == 2
    assert result.metrics.observations == 42
    assert result.metrics.matched_observations == 42
    assert result.metrics.id_switches >= 0


def test_empty_scenario_rejected():
    with pytest.raises(ValueError):
        run_scenario("empty", [])


def test_invalid_box_size_rejected():
    with pytest.raises(ValueError):
        run_scenario(
            "invalid",
            parallel_scenario(2),
            box_size=0,
        )


def test_custom_tracker():
    tracker = ObjectTracker(
        iou_threshold=0.3,
        max_missing_frames=5,
        min_confirmed_hits=2,
    )

    result = run_scenario(
        "custom",
        parallel_scenario(2),
        tracker=tracker,
    )

    assert result.objects == 2
