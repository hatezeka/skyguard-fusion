from app.detection.advanced_tracking_adapter import (
    AdvancedTrackingAdapter,
)
from app.detection.engine import Detection, BoundingBox


def detection(x, object_class="drone"):
    return Detection(
        object_class=object_class,
        confidence=0.95,
        bbox=BoundingBox(
            x=x,
            y=100.0,
            width=20.0,
            height=20.0,
        ),
    )


def test_adapter_returns_confirmed_tracks():
    adapter = AdvancedTrackingAdapter()

    result = adapter.update(
        frame_index=0,
        detections=[
            detection(20.0),
            detection(200.0),
        ],
    )

    assert len(result.confirmed) == 2
    assert len(result.ambiguous) == 0

    assert {
        observation.track_id
        for observation in result.confirmed
    } == {1, 2}

    assert all(
        observation.state == "confirmed"
        for observation in result.confirmed
    )


def test_adapter_supports_unknown_class():
    adapter = AdvancedTrackingAdapter()

    result = adapter.update(
        frame_index=0,
        detections=[
            detection(50.0, "unknown"),
        ],
    )

    assert len(result.confirmed) == 1
    assert result.confirmed[0].detection.object_class == "unknown"


def test_adapter_reset_restarts_track_ids():
    adapter = AdvancedTrackingAdapter()

    adapter.update(
        frame_index=0,
        detections=[detection(30.0)],
    )

    adapter.reset()

    result = adapter.update(
        frame_index=0,
        detections=[detection(40.0)],
    )

    assert result.confirmed[0].track_id == 1


def test_adapter_does_not_create_tracks_without_detections():
    adapter = AdvancedTrackingAdapter()

    result = adapter.update(
        frame_index=0,
        detections=[],
    )

    assert result.confirmed == []
    assert result.ambiguous == []
