import pytest

from app.detection.kalman_cv import KalmanCV2D
from app.detection.global_association_v2 import (
    associate_probabilistic,
)


def tracks_at(*positions):
    tracks = [KalmanCV2D(x, y) for x, y in positions]
    for track in tracks:
        track.predict()
    return tracks


def test_empty_tracks():
    result = associate_probabilistic([], [(10, 10)])
    assert result.unmatched_detections == (0,)


def test_empty_detections():
    result = associate_probabilistic(
        tracks_at((10, 10)), []
    )
    assert result.unmatched_tracks == (0,)


def test_separated_tracks():
    result = associate_probabilistic(
        tracks_at((10, 10), (100, 100)),
        [(100, 100), (10, 10)],
    )

    pairs = {
        (m.track_index, m.detection_index)
        for m in result.matches
    }

    assert pairs == {(0, 1), (1, 0)}


def test_identical_tracks_are_ambiguous():
    result = associate_probabilistic(
        tracks_at((50, 50), (50, 50)),
        [(50, 50), (50, 50)],
    )

    assert len(result.matches) == 2
    assert result.globally_ambiguous
    assert result.cost_gap == pytest.approx(0)


def test_one_detection_two_tracks_is_ambiguous():
    result = associate_probabilistic(
        tracks_at((50, 50), (50, 50)),
        [(50, 50)],
    )

    assert len(result.matches) == 1
    assert len(result.unmatched_tracks) == 1
    assert result.globally_ambiguous


def test_one_track_two_detections_is_ambiguous():
    result = associate_probabilistic(
        tracks_at((50, 50)),
        [(50, 50), (50, 50)],
    )

    assert len(result.matches) == 1
    assert len(result.unmatched_detections) == 1
    assert result.globally_ambiguous


def test_far_detections_unmatched():
    result = associate_probabilistic(
        tracks_at((10, 10)),
        [(500, 500)],
    )

    assert result.matches == ()
    assert result.unmatched_tracks == (0,)
    assert result.unmatched_detections == (0,)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        associate_probabilistic(
            [], [], detection_probability=1.0
        )

    with pytest.raises(ValueError):
        associate_probabilistic(
            [], [], clutter_density=0
        )


def test_invalid_coordinates():
    with pytest.raises(ValueError):
        associate_probabilistic(
            tracks_at((0, 0)),
            [(float("nan"), 0)],
        )


def test_no_filter_mutation():
    tracks = tracks_at((20, 20))
    before_state = tracks[0].state.copy()
    before_cov = tracks[0].P.copy()

    associate_probabilistic(tracks, [(20, 20)])

    assert (tracks[0].state == before_state).all()
    assert (tracks[0].P == before_cov).all()


def test_unmatched_optimum_has_feasible_alternative():
    result = associate_probabilistic(
        tracks_at((50, 50)),
        [(50, 50)],
        detection_probability=0.9,
        clutter_density=1.0,
    )

    assert result.matches == ()
    assert result.alternative_cost is not None
    assert result.alternative_cost > result.best_cost
    assert result.cost_gap is not None
    assert result.cost_gap > 0


def test_no_feasible_edge_has_no_alternative():
    result = associate_probabilistic(
        tracks_at((10, 10)),
        [(500, 500)],
    )

    assert result.matches == ()
    assert result.alternative_cost is None
    assert result.cost_gap is None


def test_selected_matching_has_alternative():
    result = associate_probabilistic(
        tracks_at((50, 50)),
        [(50, 50)],
        detection_probability=0.9,
        clutter_density=0.0001,
    )

    assert len(result.matches) == 1
    assert result.alternative_cost is not None
    assert result.alternative_cost > result.best_cost
