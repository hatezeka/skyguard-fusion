import pytest

from app.detection.engine import BoundingBox, Detection
from app.detection.bytetrack_adapter import ByteTrackAdapter


def make_detection(x, y=20.0, label="drone"):
    return Detection(
        object_class=label,
        confidence=0.95,
        bbox=BoundingBox(
            x=x,
            y=y,
            width=20,
            height=20,
        ),
    )


def test_empty_frame():
    tracker = ByteTrackAdapter()
    assert tracker.update(0, []) == []


def test_single_drone_stable_id():
    tracker = ByteTrackAdapter()

    first = tracker.update(0, [make_detection(20)])
    second = tracker.update(1, [make_detection(22)])

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_two_drones_have_different_ids():
    tracker = ByteTrackAdapter()

    result = tracker.update(
        0,
        [make_detection(20), make_detection(120)],
    )

    assert len(result) == 2
    assert result[0].track_id != result[1].track_id


def test_detection_index_mapping():
    tracker = ByteTrackAdapter()

    result = tracker.update(
        0,
        [make_detection(20), make_detection(120)],
    )

    assert {item.detection_index for item in result} == {0, 1}


def test_rejects_non_increasing_frames():
    tracker = ByteTrackAdapter()
    tracker.update(0, [])

    with pytest.raises(ValueError):
        tracker.update(0, [])


def test_rejects_skipped_frames():
    tracker = ByteTrackAdapter()
    tracker.update(0, [])

    with pytest.raises(ValueError):
        tracker.update(2, [])


def test_reset():
    tracker = ByteTrackAdapter()
    tracker.update(0, [make_detection(20)])
    tracker.reset()

    result = tracker.update(0, [make_detection(20)])
    assert len(result) == 1
