"""41.2: no device required; exact evidence and session isolation invariants."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pytest

from app.alerts.live_advisory import LiveAdvisoryBridge

SID_A = 'a' * 32
SID_B = 'b' * 32


def sample(frame=1, *, sid=SID_A, camera='cam_1', cls='drone',
           state='confirmed', confidence=0.96, tid=7, ambiguous=None):
    return {
        'state': 'running', 'session_id': sid, 'camera_id': camera,
        'frame_index': frame,
        'tracks': [{'track_id': tid, 'class': cls, 'state': state,
                    'frame_index': frame, 'confidence': confidence}],
        'ambiguous_tracks': [] if ambiguous is None else ambiguous,
    }


def evaluate(bridge, frame, **kwargs):
    item = sample(frame, **kwargs)
    return bridge.evaluate(lambda: item)


def test_early_frames_do_not_alert():
    bridge = LiveAdvisoryBridge()
    assert evaluate(bridge, 1)['alerts'] == []
    assert evaluate(bridge, 2)['alerts'] == []
    assert evaluate(bridge, 3)['alerts'][0]['severity'] == 'attention'


def test_same_frame_retry_is_identical_and_does_not_advance_streak():
    b = LiveAdvisoryBridge()
    first = evaluate(b, 1)
    assert evaluate(b, 1) == first
    second = evaluate(b, 2)
    assert second['alerts'] == []
    third = evaluate(b, 3)
    assert len(third['alerts']) == 1
    assert evaluate(b, 3) == third
    assert len(third['alerts'][0]['event_id']) == 64
    assert evaluate(b, 4)['alerts'] == []


def test_session_restart_reuses_track_id_but_resets_streak():
    b = LiveAdvisoryBridge()
    for frame in (1, 2, 3):
        before = evaluate(b, frame)
    assert len(before['alerts']) == 1
    for frame in (1, 2, 3):
        after = evaluate(b, frame, sid=SID_B)
        if frame < 3:
            assert not after['alerts']
    assert before['alerts'][0]['event_id'] != after['alerts'][0]['event_id']


def test_birds_do_not_create_alerts():
    b = LiveAdvisoryBridge()
    for frame in range(1, 6):
        assert not evaluate(b, frame, cls='bird')['alerts']


def test_ambiguous_association_suppresses_alerts():
    b = LiveAdvisoryBridge()
    ambiguous = [{'state': 'ambiguous', 'frame_index': 1, 'candidate_track_ids':[7]}]
    for frame in (1, 2, 3):
        ambiguous[0]['frame_index'] = frame
        assert not evaluate(b, frame, ambiguous=ambiguous)['alerts']


def test_model_advisory_does_not_claim_gps_range_or_breach():
    b = LiveAdvisoryBridge()
    for frame in (1, 2, 3):
        result = evaluate(b, frame)
    assert result['target_gps_available'] is False
    assert result['facility_distance_available'] is False
    assert result['geofence_breach_verified'] is False
    assert result['automated_response_allowed'] is False
    assert result['alert_archive_persistent'] is False
    assert result['sampled_frames_only'] is True
    assert result['alerts'][0]['target_range_m'] is None
    assert result['alerts'][0]['ground_truth_verified'] is False


def test_inactive_session_clears_streaks():
    b = LiveAdvisoryBridge()
    evaluate(b, 1)
    evaluate(b, 2)
    with pytest.raises(ValueError, match='No running'):
        b.evaluate(lambda: {'state': 'stopped'})
    assert evaluate(b, 3)['alerts'] == []


def test_late_restart_between_two_reads_is_rejected():
    b = LiveAdvisoryBridge()
    observations = iter([sample(1), sample(1, sid=SID_B)])
    with pytest.raises(ValueError, match='session changed'):
        b.evaluate(lambda: next(observations))
    assert evaluate(b, 2, sid=SID_B)['alerts'] == []


def test_frame_advance_during_request_retains_sample_identity():
    b = LiveAdvisoryBridge()
    items = iter([sample(1), sample(2)])
    result = b.evaluate(lambda: next(items))
    assert result['frame_index'] == 1
    assert result['latest_frame_index_at_check'] == 2
    assert result['alerts'] == []


def test_stale_frame_rejected():
    b = LiveAdvisoryBridge()
    evaluate(b, 4)
    with pytest.raises(ValueError, match='out-of-order'):
        evaluate(b, 2)


def test_conflicting_same_frame_rejected():
    b = LiveAdvisoryBridge()
    evaluate(b, 1)
    with pytest.raises(ValueError, match='conflicting'):
        evaluate(b, 1, confidence=0.93)


def test_unregistered_camera_does_not_alert():
    b = LiveAdvisoryBridge()
    for frame in (1, 2, 3, 4):
        result = evaluate(b, frame, camera=None)
    assert not result['alerts']


def test_concurrent_replay_is_idempotent():
    b = LiveAdvisoryBridge()
    for frame in (1, 2):
        evaluate(b, frame)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: evaluate(b, 3), range(40)))
    assert all(results[0] == result for result in results)
    assert len(results[0]['alerts']) == 1


def test_external_cannot_mutate_bridge_replay():
    b = LiveAdvisoryBridge()
    for frame in (1, 2):
        evaluate(b, frame)
    original = evaluate(b, 3)
    mutated = deepcopy(original)
    mutated['alerts'][0]['event_id'] = 'spoofed'
    assert evaluate(b, 3) == original


@dataclass(frozen=True)
class DataclassSnapshot:
    state: str
    session_id: str
    camera_id: str
    frame_index: int
    tracks: list
    ambiguous_tracks: list


def test_dataclass_snapshot_supported():
    b = LiveAdvisoryBridge()
    raw = sample()
    obj = DataclassSnapshot(**raw)
    assert b.evaluate(lambda: obj)['frame_index'] == 1


def test_invalid_snapshot_type_rejected():
    with pytest.raises(ValueError, match='dataclass or mapping'):
        LiveAdvisoryBridge().evaluate(lambda: 42)
