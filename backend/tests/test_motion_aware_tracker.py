import pytest

from app.detection.engine import BoundingBox, Detection
from app.detection.motion_aware_tracker import (
    MotionAwareTracker,
    _hungarian,
)


def detection(x, y=20, label="drone"):
    return Detection(
        object_class=label,
        confidence=0.95,
        bbox=BoundingBox(x=x, y=y, width=20, height=20),
    )


def test_hungarian_global_optimum():
    assert _hungarian([[9, 2], [1, 8]]) == [1, 0]


def test_hungarian_rectangular():
    assert _hungarian([[10, 1, 5], [2, 8, 9]]) == [1, 0]


def test_single_drone_stable():
    tracker = MotionAwareTracker()
    ids = [
        tracker.update(i, [detection(20 + i * 15)])[0].track_id
        for i in range(10)
    ]
    assert len(set(ids)) == 1


def test_fast_two_drones():
    tracker = MotionAwareTracker()
    ids_a = []
    ids_b = []

    for i in range(11):
        output = tracker.update(
            i,
            [
                detection(20 + i * 20),
                detection(220 - i * 20),
            ],
        )

        by_index = {
            obs.detection_index: obs.track_id
            for obs in output
        }

        assert len(by_index) == 2
        ids_a.append(by_index[0])
        ids_b.append(by_index[1])

    assert len(set(ids_a)) == 1
    assert len(set(ids_b)) == 1
    assert ids_a[0] != ids_b[0]


def test_order_reversal():
    tracker = MotionAwareTracker()
    original_ids = None

    for i in range(8):
        detections = [
            detection(20 + i * 10),
            detection(200 - i * 10),
        ]
        if i % 2:
            detections.reverse()

        output = tracker.update(i, detections)
        identity_ids = {}

        for obs in output:
            x = obs.detection.bbox.x
            identity = (
                "A" if x == 20 + i * 10 else "B"
            )
            identity_ids[identity] = obs.track_id

        if original_ids is None:
            original_ids = identity_ids
        else:
            assert identity_ids == original_ids


def test_short_occlusion():
    tracker = MotionAwareTracker(max_missing_frames=4)
    first = tracker.update(0, [detection(20)])[0].track_id
    tracker.update(1, [detection(30)])
    tracker.update(2, [])
    tracker.update(3, [])

    recovered = tracker.update(
        4, [detection(60)]
    )[0].track_id

    assert recovered == first


def test_class_separation():
    tracker = MotionAwareTracker()
    drone_id = tracker.update(
        0, [detection(20, label="drone")]
    )[0].track_id

    bird_id = tracker.update(
        1, [detection(21, label="bird")]
    )[0].track_id

    assert drone_id != bird_id


def test_reset():
    tracker = MotionAwareTracker()
    tracker.update(0, [detection(20)])
    tracker.reset()
    assert tracker.update(
        0, [detection(20)]
    )[0].track_id == 1


def test_reject_skipped_frames():
    tracker = MotionAwareTracker()
    tracker.update(0, [])
    with pytest.raises(ValueError):
        tracker.update(2, [])


def test_invalid_config():
    with pytest.raises(ValueError):
        MotionAwareTracker(max_distance=0)
    with pytest.raises(ValueError):
        MotionAwareTracker(max_missing_frames=-1)
    with pytest.raises(ValueError):
        MotionAwareTracker(velocity_alpha=2)


def test_occlusion_does_not_steal_visible_identity():
    from app.tracking_validation.scenarios import crossing_scenario

    tracker = MotionAwareTracker()
    original = crossing_scenario()
    initial_ids = None

    for frame in original:
        objects = [
            obj for obj in frame.objects
            if not (
                obj.identity == "DRONE_A"
                and 8 <= frame.frame_index <= 11
            )
        ]

        detections = [
            detection(obj.x, obj.y, obj.object_class)
            for obj in objects
        ]

        output = tracker.update(frame.frame_index, detections)

        assignments = {
            objects[obs.detection_index].identity: obs.track_id
            for obs in output
        }

        if initial_ids is None:
            initial_ids = assignments.copy()

        assert assignments.get("DRONE_B") == initial_ids["DRONE_B"]

        if "DRONE_A" in assignments:
            assert assignments["DRONE_A"] == initial_ids["DRONE_A"]
