import pytest

from app.detection.kalman_cv import KalmanCV2D
from app.detection.global_association import associate_global


def tracks_at(*positions):
    tracks = [KalmanCV2D(x, y) for x, y in positions]
    for track in tracks:
        track.predict()
    return tracks


def test_empty_tracks():
    result = associate_global([], [(10, 10)])
    assert result.unmatched_detections == (0,)
    assert result.matches == ()


def test_empty_detections():
    result = associate_global(tracks_at((10, 10)), [])
    assert result.unmatched_tracks == (0,)


def test_two_separated_tracks():
    tracks = tracks_at((10, 10), (100, 100))
    result = associate_global(
        tracks, [(100, 100), (10, 10)]
    )

    assert {
        (match.track_index, match.detection_index)
        for match in result.matches
    } == {(0, 1), (1, 0)}

    assert not result.globally_ambiguous


def test_far_detection_remains_unmatched():
    tracks = tracks_at((10, 10))
    result = associate_global(tracks, [(500, 500)])

    assert result.matches == ()
    assert result.unmatched_tracks == (0,)
    assert result.unmatched_detections == (0,)


def test_identical_tracks_are_ambiguous():
    tracks = tracks_at((50, 50), (50, 50))
    result = associate_global(
        tracks, [(50, 50), (50, 50)]
    )

    assert len(result.matches) == 2
    assert result.globally_ambiguous
    assert result.cost_gap == pytest.approx(0)


def test_single_track_near_detection():
    tracks = tracks_at((20, 20))
    result = associate_global(tracks, [(21, 20)])

    assert len(result.matches) == 1
    assert result.matches[0].track_index == 0
    assert result.matches[0].detection_index == 0


def test_filter_is_not_mutated():
    tracks = tracks_at((20, 20))
    old_state = tracks[0].state.copy()
    old_covariance = tracks[0].P.copy()

    associate_global(tracks, [(21, 20)])

    assert (tracks[0].state == old_state).all()
    assert (tracks[0].P == old_covariance).all()


def test_invalid_configuration():
    with pytest.raises(ValueError):
        associate_global([], [], gate_threshold=-1)

    with pytest.raises(ValueError):
        associate_global([], [], ambiguity_gap=-1)


def test_invalid_detection():
    with pytest.raises(ValueError):
        associate_global(
            tracks_at((0, 0)),
            [(float("nan"), 0)],
        )

