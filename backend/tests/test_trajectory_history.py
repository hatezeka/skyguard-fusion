from datetime import datetime, timedelta, timezone

import pytest

from app.detection.engine import Detection, BoundingBox
from app.detection.advanced_tracking_adapter import AdvancedTrackObservation
from app.detection.trajectory_history import TrajectoryHistory


T0 = datetime(2026, 11, 20, tzinfo=timezone.utc)


def observation(track_id, frame, x, obj_class="drone"):
    detection = Detection(
        object_class=obj_class,
        confidence=0.95,
        bbox=BoundingBox(
            x=x,
            y=100.0,
            width=20.0,
            height=20.0,
        ),
    )

    return AdvancedTrackObservation(
        track_id=track_id,
        detection=detection,
        frame_index=frame,
    )


def test_independent_track_histories():
    engine = TrajectoryHistory()

    engine.update(
        [observation(1, 0, 10), observation(2, 0, 200)],
        T0,
    )

    assert engine.track_count == 2
    assert engine.get_history(1)[0].x_px == 20
    assert engine.get_history(2)[0].x_px == 210


def test_velocity_uses_elapsed_seconds():
    engine = TrajectoryHistory()

    engine.update([observation(1, 0, 10)], T0)
    engine.update(
        [observation(1, 1, 30)],
        T0 + timedelta(seconds=2),
    )

    point = engine.get_history(1)[-1]

    assert point.velocity_x_px_s == pytest.approx(10)
    assert point.velocity_y_px_s == pytest.approx(0)
    assert point.speed_px_s == pytest.approx(10)


def test_unknown_class_is_preserved():
    engine = TrajectoryHistory()

    engine.update(
        [observation(1, 0, 10, "unknown")],
        T0,
    )

    assert engine.get_history(1)[0].object_class == "unknown"


def test_history_is_bounded():
    engine = TrajectoryHistory(max_points_per_track=3)

    for frame in range(5):
        engine.update(
            [observation(1, frame, frame * 10)],
            T0 + timedelta(seconds=frame),
        )

    history = engine.get_history(1)

    assert len(history) == 3
    assert history[0].frame_index == 2
    assert history[-1].frame_index == 4


def test_frame_index_must_increase():
    engine = TrajectoryHistory()
    engine.update([observation(1, 0, 10)], T0)

    with pytest.raises(ValueError):
        engine.update(
            [observation(1, 0, 20)],
            T0 + timedelta(seconds=1),
        )


def test_timestamp_must_be_timezone_aware():
    engine = TrajectoryHistory()

    with pytest.raises(ValueError):
        engine.update(
            [observation(1, 0, 10)],
            datetime(2026, 11, 20),
        )


def test_duplicate_track_ids_are_rejected():
    engine = TrajectoryHistory()

    with pytest.raises(ValueError):
        engine.update(
            [
                observation(1, 0, 10),
                observation(1, 0, 100),
            ],
            T0,
        )

    assert engine.track_count == 0


def test_capacity_is_not_silently_exceeded():
    engine = TrajectoryHistory(max_tracks=1)

    engine.update([observation(1, 0, 10)], T0)

    with pytest.raises(ValueError):
        engine.update(
            [observation(2, 0, 100)],
            T0 + timedelta(seconds=1),
        )

    assert engine.track_count == 1


def test_remove_and_reset():
    engine = TrajectoryHistory()
    engine.update([observation(1, 0, 10)], T0)

    assert engine.remove_track(1)
    assert not engine.remove_track(1)

    engine.update(
        [observation(2, 0, 20)],
        T0 + timedelta(seconds=1),
    )

    engine.reset()
    assert engine.track_count == 0


def test_ambiguous_observations_are_not_accepted():
    engine = TrajectoryHistory()

    class Ambiguous:
        detection = observation(1, 0, 10).detection
        frame_index = 0

    with pytest.raises(AttributeError):
        engine.update([Ambiguous()], T0)

    assert engine.track_count == 0
