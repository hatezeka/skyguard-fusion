from app.detection.engine import BoundingBox, Detection
from app.detection.motion_aware_v2 import MotionAwareTrackerV2


def make_detection(object_class, x, y=40):
    return Detection(
        object_class=object_class,
        confidence=0.9,
        bbox=BoundingBox(float(x), float(y), 20.0, 20.0),
    )


def test_unknown_to_drone_preserves_identity():
    tracker = MotionAwareTrackerV2()

    first = tracker.update(
        0, [make_detection("unknown", 10)]
    )
    second = tracker.update(
        1, [make_detection("drone", 15)]
    )

    assert len(first.confirmed) == 1
    assert len(second.confirmed) == 1
    assert (
        second.confirmed[0].track_id
        == first.confirmed[0].track_id
    )


def test_drone_to_unknown_preserves_identity():
    tracker = MotionAwareTrackerV2()

    first = tracker.update(
        0, [make_detection("drone", 10)]
    )
    second = tracker.update(
        1, [make_detection("unknown", 15)]
    )

    assert len(first.confirmed) == 1
    assert len(second.confirmed) == 1
    assert (
        second.confirmed[0].track_id
        == first.confirmed[0].track_id
    )


def test_unknown_to_bird_preserves_identity():
    tracker = MotionAwareTrackerV2()

    first = tracker.update(
        0, [make_detection("unknown", 10)]
    )
    second = tracker.update(
        1, [make_detection("bird", 15)]
    )

    assert len(first.confirmed) == 1
    assert len(second.confirmed) == 1
    assert (
        second.confirmed[0].track_id
        == first.confirmed[0].track_id
    )


def test_distant_object_does_not_inherit_identity():
    tracker = MotionAwareTrackerV2(max_distance=30)

    first = tracker.update(
        0, [make_detection("unknown", 10)]
    )
    second = tracker.update(
        1, [make_detection("drone", 150)]
    )

    assert len(first.confirmed) == 1
    assert len(second.confirmed) == 1
    assert (
        second.confirmed[0].track_id
        != first.confirmed[0].track_id
    )


def test_same_class_motion_preserves_identity():
    tracker = MotionAwareTrackerV2()

    first = tracker.update(
        0, [make_detection("drone", 10)]
    )
    second = tracker.update(
        1, [make_detection("drone", 15)]
    )

    assert (
        second.confirmed[0].track_id
        == first.confirmed[0].track_id
    )
