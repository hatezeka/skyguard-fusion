import numpy as np
import pytest

from app.video import stereo_capture
from app.video.source import VideoFrame


def sample(index, milliseconds):
    return stereo_capture.Captured(VideoFrame(index, np.empty((0, 0, 3)), 0, 0),
                                   milliseconds * 1_000_000, 1.0)


def test_stereo_pair_discards_stale_and_excessively_skewed_frames(monkeypatch):
    monkeypatch.setattr(stereo_capture, "perf_counter_ns", lambda: 1_000_000_000)
    capture = stereo_capture.StereoCapture(0, 1)
    capture.queues[0].extend([sample(1, 800), sample(2, 880), sample(3, 980)])
    capture.queues[1].append(sample(4, 970))
    left, right = capture.next_pair(timeout_s=0)
    assert (left.frame.index, right.frame.index) == (3, 4)
    assert capture.report()["left"]["unpaired_drops"] == 2
    with pytest.raises(TimeoutError):
        capture.next_pair(timeout_s=0)


def test_synthetic_stereo_match_enriches_one_id_without_publishing_range():
    from types import SimpleNamespace
    from app.detection.engine import BoundingBox, Detection
    from app.detection.stereo_observation import stereo_observations

    left = np.random.default_rng(42).integers(0, 256, (160, 220, 3), dtype=np.uint8)
    right = np.roll(left, -12, axis=1)
    primary = Detection("drone", .9, BoundingBox(60, 25, 100, 110))
    peer = Detection("drone", .9, BoundingBox(48, 25, 100, 110))
    unknown = Detection("unknown", .5, BoundingBox(10, 10, 25, 25))

    def synthetic_geometry(a, b, *, max_epipolar_px):
        if abs(a[1] - b[1]) > max_epipolar_px or abs(a[0] - b[0] - 12) > 1:
            raise ValueError("Mismatch")

    profile = SimpleNamespace(triangulate_rectified=synthetic_geometry)
    tracks = [SimpleNamespace(track_id=17, detection=primary),
              SimpleNamespace(track_id=18, detection=unknown)]
    result, = stereo_observations(left, right, tracks, [peer, unknown], profile)
    assert result.track_id == 17 and result.right_bbox is not None
    assert result.matched_points >= 4
    assert result.range_status == "CALIBRATION_SUSPECT"
    assert result.xyz_m is None and result.range_m is None and result.wgs84 is None

    unmatched, = stereo_observations(left, right, tracks, [unknown], profile)
    assert unmatched.track_id == 17 and unmatched.right_bbox is None
    assert unmatched.range_status == "RANGE_UNAVAILABLE"
