import pytest

from app.detection.kalman_cv import KalmanCV2D
from app.detection.identity_hypotheses import generate_hypotheses
from app.detection.global_association_v2 import associate_probabilistic


def predicted(*positions):
    tracks = [KalmanCV2D(x, y) for x, y in positions]
    for track in tracks:
        track.predict()
    return tracks


def test_empty():
    hypotheses = generate_hypotheses([], [])
    assert len(hypotheses) == 1
    assert hypotheses[0].matches == ()
    assert hypotheses[0].cost == 0


def test_separated_tracks():
    tracks = predicted((10, 10), (100, 100))
    detections = [(100, 100), (10, 10)]

    hypotheses = generate_hypotheses(tracks, detections)

    assert hypotheses[0].matches == ((0, 1), (1, 0))


def test_identical_tracks_have_equal_hypotheses():
    tracks = predicted((50, 50), (50, 50))
    detections = [(50, 50), (50, 50)]

    hypotheses = generate_hypotheses(tracks, detections)

    assert hypotheses[0].matches != hypotheses[1].matches
    assert hypotheses[0].cost == pytest.approx(
        hypotheses[1].cost
    )


def test_top_k_limit():
    tracks = predicted((50, 50), (50, 50))
    detections = [(50, 50), (50, 50)]

    hypotheses = generate_hypotheses(
        tracks, detections, max_hypotheses=3
    )

    assert len(hypotheses) == 3
    assert all(
        hypotheses[i].cost <= hypotheses[i + 1].cost
        for i in range(len(hypotheses) - 1)
    )


def test_best_matches_existing_association():
    tracks = predicted((10, 10), (100, 100))
    detections = [(11, 10), (99, 100)]

    hypotheses = generate_hypotheses(tracks, detections)
    baseline = associate_probabilistic(tracks, detections)

    assert hypotheses[0].cost == pytest.approx(
        baseline.best_cost
    )


def test_unmatched_hypothesis_is_preserved():
    hypotheses = generate_hypotheses(
        predicted((50, 50)),
        [(50, 50)],
        clutter_density=1.0,
    )

    assert hypotheses[0].matches == ()
    assert any(h.matches == ((0, 0),) for h in hypotheses)


def test_invalid_limits():
    with pytest.raises(ValueError):
        generate_hypotheses([], [], max_hypotheses=0)

    with pytest.raises(ValueError):
        generate_hypotheses(
            predicted(*[(0, 0)] * 5),
            [],
        )


def test_filters_not_mutated():
    tracks = predicted((20, 20))
    state_before = tracks[0].state.copy()
    covariance_before = tracks[0].P.copy()

    generate_hypotheses(tracks, [(20, 20)])

    assert (tracks[0].state == state_before).all()
    assert (tracks[0].P == covariance_before).all()
