"""Phase 41.1 strict evidence-only advisory policy tests."""
from __future__ import annotations

from copy import deepcopy
import math
import pytest

from app.alerts.risk_engine import AdvisoryPolicy, EvidenceAlertEngine

SID = "a" * 32
SID2 = "b" * 32


def track(tid=1, label="drone", confidence=0.97, state="confirmed", frame=1):
    return {"track_id": tid, "class": label, "confidence": confidence,
            "state": state, "frame_index": frame}


def snap(frame=1, tracks=None, sid=SID, camera="cam_1", ambiguous=None):
    return {"state": "running", "session_id": sid, "camera_id": camera,
            "frame_index": frame,
            "tracks": [track(frame=frame)] if tracks is None else tracks,
            "ambiguous_tracks": [] if ambiguous is None else ambiguous}


def ambiguity(frame, ids):
    return {"state": "ambiguous", "frame_index": frame, "candidate_track_ids": ids}


def advance(engine, *frames):
    return [engine.evaluate(snap(n)) for n in frames]


def test_policy_defaults():
    assert AdvisoryPolicy().min_observations == 3
    assert AdvisoryPolicy().min_confidence == 0.85


@pytest.mark.parametrize("kwargs", [
    {"min_observations": 1}, {"min_observations": 0}, {"min_observations": 101},
    {"min_observations": 2.0}, {"min_confidence": False},
    {"min_confidence": float("nan")}, {"min_confidence": 1.1},
])
def test_invalid_policies(kwargs):
    with pytest.raises(ValueError): AdvisoryPolicy(**kwargs)


def test_wait_for_repeated_independent_frames():
    e = EvidenceAlertEngine()
    first, second, third = advance(e, 1, 2, 3)
    assert first["alerts"] == second["alerts"] == []
    assert [a["track_id"] for a in third["alerts"]] == [1]
    assert third["alerts"][0]["severity"] == "attention"


def test_deduplicated_advisory_and_idempotent_replay():
    e = EvidenceAlertEngine()
    advance(e, 1, 2)
    msg = e.evaluate(snap(3))
    assert len(msg["alerts"]) == 1
    replay = e.evaluate(snap(3))
    assert replay == msg
    replay["alerts"].clear()
    assert len(e.evaluate(snap(3))["alerts"]) == 1
    assert e.evaluate(snap(4))["alerts"] == []


def test_same_frame_conflicting_snapshot_rejected():
    e = EvidenceAlertEngine()
    e.evaluate(snap(1))
    with pytest.raises(ValueError, match="conflicting"):
        e.evaluate(snap(1, [track(confidence=0.6)]))


def test_out_of_order_frame_rejected():
    e = EvidenceAlertEngine()
    e.evaluate(snap(3))
    with pytest.raises(ValueError, match="out-of-order"):
        e.evaluate(snap(2))


def test_new_session_resets_streak_and_reused_track_id():
    e = EvidenceAlertEngine()
    advance(e, 1, 2, 3)
    assert e.evaluate(snap(1, sid=SID2))["alerts"] == []
    assert e.evaluate(snap(2, sid=SID2))["alerts"] == []
    assert len(e.evaluate(snap(3, sid=SID2))["alerts"]) == 1


def test_inactive_track_does_not_accumulate():
    e = EvidenceAlertEngine()
    e.evaluate(snap(1))
    e.evaluate(snap(2, [track(frame=1)]))
    assert e.evaluate(snap(3))["alerts"] == []


def test_bird_unknown_never_trigger_even_high_confidence():
    for label in ("bird", "unknown", "airplane", "helicopter"):
        e = EvidenceAlertEngine()
        for frame in range(1, 6):
            assert not e.evaluate(snap(frame, [track(label=label, frame=frame)]))["alerts"]


def test_model_confidence_floor():
    e = EvidenceAlertEngine()
    for frame in range(1, 6):
        assert not e.evaluate(snap(frame, [track(confidence=.84, frame=frame)]))["alerts"]


def test_ambiguous_candidate_suppresses_and_resets_streak():
    e = EvidenceAlertEngine()
    e.evaluate(snap(1))
    second = e.evaluate(snap(2, ambiguous=[ambiguity(2, [1, 9])]))
    assert second["alerts"] == []
    assert second["observations"][0]["model_evidence_status"] == "ambiguous_association"
    assert e.evaluate(snap(3))["alerts"] == []
    assert e.evaluate(snap(4))["alerts"] == []
    assert len(e.evaluate(snap(5))["alerts"]) == 1


def test_two_tracks_are_independent_when_crossing_pixel_paths():
    e = EvidenceAlertEngine()
    for frame in (1, 2, 3):
        result = e.evaluate(snap(frame, [track(11, frame=frame), track(22, frame=frame)]))
    assert {alert["track_id"] for alert in result["alerts"]} == {11, 22}
    assert all(alert["target_gps"] is None for alert in result["alerts"])


def test_unknown_camera_never_triggers_advisory():
    e = EvidenceAlertEngine()
    for frame in range(1, 5):
        result = e.evaluate(snap(frame, camera=None))
    assert result["alerts"] == []
    assert result["observations"][0]["model_evidence_status"] == "unregistered_camera"


def test_stop_or_idle_not_accepted_as_running():
    for state in ("idle", "starting", "stopped", "error"):
        e = EvidenceAlertEngine()
        item = snap(1)
        item["state"] = state
        with pytest.raises(ValueError, match="running"):
            e.evaluate(item)


def test_expected_identity_mismatch_fails_closed():
    e = EvidenceAlertEngine()
    with pytest.raises(ValueError, match="session identity"):
        e.evaluate(snap(1), expected_session_id=SID2)
    with pytest.raises(ValueError, match="camera identity"):
        e.evaluate(snap(1), expected_camera_id="cam_2")


@pytest.mark.parametrize("broken", [
    {"session_id": "bad"}, {"frame_index": -1}, {"frame_index": True},
    {"tracks": [track(), track()]}, {"tracks": [track(confidence=float("nan"))]},
    {"tracks": [track(confidence=True)]}, {"tracks": [track(frame=2)]},
    {"tracks": [track(label="bad class!")]},
    {"ambiguous_tracks": [ambiguity(4, [1])]},
])
def test_invalid_snapshots_rejected(broken):
    item = snap(1)
    item.update(broken)
    with pytest.raises(ValueError): EvidenceAlertEngine().evaluate(item)


def test_rejects_oversized_payload():
    e = EvidenceAlertEngine()
    with pytest.raises(ValueError, match="oversized"):
        e.evaluate(snap(1, [track(i, frame=1) for i in range(513)]))


def test_no_geographical_or_automated_response_claims():
    e = EvidenceAlertEngine()
    result = advance(e, 1, 2, 3)[-1]
    assert result["target_gps_available"] is False
    assert result["facility_distance_available"] is False
    assert result["geofence_breach_verified"] is False
    assert result["automated_response_allowed"] is False
    assert result["alerts"][0]["facility_breach_verified"] is False
    assert result["alerts"][0]["ground_truth_verified"] is False
    assert result["alerts"][0]["target_range_m"] is None


def test_reset_explicit():
    e = EvidenceAlertEngine()
    advance(e, 1, 2, 3)
    e.reset()
    assert e.evaluate(snap(1))["alerts"] == []


def test_alert_reissued_only_after_fresh_new_streak():
    e = EvidenceAlertEngine()
    advance(e, 1, 2, 3)
    e.evaluate(snap(4, [track(label="bird", frame=4)]))
    for frame in (5, 6):
        assert e.evaluate(snap(frame))["alerts"] == []
    assert len(e.evaluate(snap(7))["alerts"]) == 1


def test_custom_policy():
    e = EvidenceAlertEngine(AdvisoryPolicy(min_observations=2, min_confidence=.9))
    assert not e.evaluate(snap(1))["alerts"]
    assert len(e.evaluate(snap(2))["alerts"]) == 1


def test_no_pollution_of_user_input():
    item = snap(1)
    original = deepcopy(item)
    EvidenceAlertEngine().evaluate(item)
    assert item == original
