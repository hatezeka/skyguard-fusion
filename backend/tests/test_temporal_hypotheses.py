import pytest

from app.detection.temporal_hypotheses import (
    TemporalHypothesisManager,
)


def test_separated_tracks():
    manager = TemporalHypothesisManager(
        [(10, 10), (100, 100)]
    )

    result = manager.update(
        0, [(10, 10), (100, 100)]
    )

    assert result.retained_hypotheses >= 1
    assert result.frame_index == 0


def test_identical_tracks_remain_unconfirmed():
    manager = TemporalHypothesisManager(
        [(50, 50), (50, 50)]
    )

    result = manager.update(
        0, [(50, 50), (50, 50)]
    )

    assert result.assignments == ()
    assert set(result.ambiguous_tracks) == {0, 1}


def test_bounded_branches():
    manager = TemporalHypothesisManager(
        [(50, 50), (50, 50)],
        max_hypotheses=4,
    )

    for frame in range(5):
        result = manager.update(
            frame, [(50, 50), (50, 50)]
        )
        assert result.retained_hypotheses <= 4


def test_consecutive_frames_required():
    manager = TemporalHypothesisManager([(10, 10)])
    manager.update(0, [(10, 10)])

    with pytest.raises(ValueError):
        manager.update(2, [(10, 10)])


def test_invalid_detection_preserves_state():
    manager = TemporalHypothesisManager([(10, 10)])
    manager.update(0, [(10, 10)])

    previous = manager.last_frame
    previous_cost = manager.branches[0].cost

    with pytest.raises(ValueError):
        manager.update(1, [(float("nan"), 10)])

    assert manager.last_frame == previous
    assert manager.branches[0].cost == previous_cost


def test_branch_filters_are_independent():
    manager = TemporalHypothesisManager(
        [(50, 50)],
        max_hypotheses=4,
    )

    manager.update(0, [(50, 50)])

    assert len(manager.branches) >= 2

    first = manager.branches[0].filters[0]
    second = manager.branches[1].filters[0]

    assert first is not second
    assert first.state is not second.state


def test_invalid_configuration():
    with pytest.raises(ValueError):
        TemporalHypothesisManager(
            [(0, 0)], max_hypotheses=0
        )

    with pytest.raises(ValueError):
        TemporalHypothesisManager(
            [(0, 0)] * 5
        )


def test_empty_detections():
    manager = TemporalHypothesisManager([(10, 10)])
    result = manager.update(0, [])

    assert result.assignments == ()
    assert result.ambiguous_tracks == (0,)


def test_clear_track_recovers_confirmation_after_pruning():
    manager = TemporalHypothesisManager(
        [(20, 20)], max_hypotheses=8
    )

    results = [
        manager.update(t, [(20 + 2*t, 20)])
        for t in range(8)
    ]

    assert any(result.truncated for result in results)
    assert all(
        result.assignments == ((0, 0),)
        for result in results
    )


def test_separated_tracks_recover_confirmation():
    manager = TemporalHypothesisManager(
        [(20, 20), (120, 120)],
        max_hypotheses=8,
    )

    results = [
        manager.update(
            t,
            [(20 + 2*t, 20), (120 - 2*t, 120)],
        )
        for t in range(8)
    ]

    assert any(result.truncated for result in results)
    assert all(
        result.assignments == ((0, 0), (1, 1))
        for result in results
    )


def test_identical_tracks_remain_ambiguous_after_pruning():
    manager = TemporalHypothesisManager(
        [(50, 50), (50, 50)],
        max_hypotheses=8,
    )

    results = [
        manager.update(t, [(50, 50), (50, 50)])
        for t in range(8)
    ]

    assert all(result.assignments == () for result in results)
    assert all(
        set(result.ambiguous_tracks) == {0, 1}
        for result in results
    )
