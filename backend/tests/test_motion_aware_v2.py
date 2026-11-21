"""Behavioral contract for MotionAwareTracker V2."""

from app.detection.engine import BoundingBox, Detection
from app.detection.motion_aware_v2 import MotionAwareTrackerV2


def detection(x, y=50):
    return Detection(
        object_class="drone",
        confidence=0.95,
        bbox=BoundingBox(
            x=x,
            y=y,
            width=20,
            height=20,
        ),
    )


def test_normal_motion_preserves_identity():
    tracker = MotionAwareTrackerV2()
    ids = []

    for frame in range(8):
        result = tracker.update(
            frame, [detection(20 + frame * 5)]
        )

        assert len(result.confirmed) == 1
        assert result.ambiguous == []
        ids.append(result.confirmed[0].track_id)

    assert len(set(ids)) == 1


def test_exact_crossing_reports_ambiguity():
    tracker = MotionAwareTrackerV2()

    for frame in range(10):
        result = tracker.update(
            frame,
            [
                detection(20 + frame * 5),
                detection(120 - frame * 5),
            ],
        )
        assert len(result.confirmed) == 2

    # At frame 10 both predicted centers coincide.
    # Only B is detected; its identity is not observable
    # from position alone.
    result = tracker.update(10, [detection(70)])

    assert len(result.confirmed) == 0
    assert len(result.ambiguous) == 1
    assert result.ambiguous[0].detection_index == 0
    assert set(result.ambiguous[0].candidate_track_ids) == {1, 2}


def test_identity_recovers_after_ambiguous_frame():
    tracker = MotionAwareTrackerV2()
    initial_ids = None

    for frame in range(10):
        result = tracker.update(
            frame,
            [
                detection(20 + frame * 5),
                detection(120 - frame * 5),
            ],
        )
        if initial_ids is None:
            initial_ids = [
                obs.track_id for obs in result.confirmed
            ]

    tracker.update(10, [detection(70)])

    result = tracker.update(
        11,
        [detection(75), detection(65)],
    )

    assert len(result.confirmed) == 2
    assert result.ambiguous == []
    assert [
        obs.track_id for obs in result.confirmed
    ] == initial_ids


def test_reset_restarts_track_ids():
    tracker = MotionAwareTrackerV2()

    tracker.update(0, [detection(20)])
    tracker.reset()

    result = tracker.update(0, [detection(20)])
    assert result.confirmed[0].track_id == 1


def test_ambiguous_detection_not_counted_as_confirmed():
    tracker = MotionAwareTrackerV2()

    for frame in range(10):
        tracker.update(
            frame,
            [
                detection(20 + frame * 5),
                detection(120 - frame * 5),
            ],
        )

    result = tracker.update(10, [detection(70)])

    assert len(result.confirmed) == 0
    assert len(result.ambiguous) == 1
    assert len(result.confirmed) + len(result.ambiguous) == 1
