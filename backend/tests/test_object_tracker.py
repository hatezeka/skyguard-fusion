import pytest

from app.detection.engine import BoundingBox, Detection
from app.detection.tracker import ObjectTracker, box_iou


def make_detection(x, y=10, label="drone"):
    return Detection(
        object_class=label,
        confidence=0.9,
        bbox=BoundingBox(
            x=float(x),
            y=float(y),
            width=20.0,
            height=20.0,
        ),
    )


def test_iou_identical_boxes():
    box = BoundingBox(10, 10, 20, 20)
    assert box_iou(box, box) == 1.0


def test_iou_disjoint_boxes():
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(30, 30, 10, 10)
    assert box_iou(first, second) == 0.0


def test_same_object_keeps_id():
    tracker = ObjectTracker()
    first = tracker.update(0, [make_detection(10)])
    second = tracker.update(1, [make_detection(12)])
    third = tracker.update(2, [make_detection(14)])

    assert first[0].track_id == second[0].track_id
    assert second[0].track_id == third[0].track_id
    assert third[0].state == "confirmed"


def test_different_objects_get_different_ids():
    tracker = ObjectTracker()
    results = tracker.update(
        0,
        [make_detection(10), make_detection(100)],
    )
    assert len({item.track_id for item in results}) == 2


def test_temporary_missing_detection_preserves_id():
    tracker = ObjectTracker(max_missing_frames=2)
    first = tracker.update(0, [make_detection(10)])
    assert tracker.update(1, []) == []
    second = tracker.update(2, [make_detection(11)])
    assert first[0].track_id == second[0].track_id


def test_expired_track_gets_new_id():
    tracker = ObjectTracker(max_missing_frames=1)
    first = tracker.update(0, [make_detection(10)])
    tracker.update(1, [])
    tracker.update(2, [])
    second = tracker.update(3, [make_detection(10)])
    assert first[0].track_id != second[0].track_id


def test_rejects_out_of_order_frames():
    tracker = ObjectTracker()
    tracker.update(3, [])
    with pytest.raises(ValueError):
        tracker.update(3, [])


def test_reset_starts_fresh():
    tracker = ObjectTracker()
    tracker.update(0, [make_detection(10)])
    tracker.reset()
    result = tracker.update(0, [make_detection(20)])
    assert result[0].track_id == 1