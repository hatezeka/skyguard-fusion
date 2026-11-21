"""Phase 41.1 — deterministic, evidence-limited camera-track advisory engine.

Advisories express *model evidence*, never a verified drone or a facility
perimeter violation. This module cannot compute range, target GPS or a
geofence breach from one optical camera. No destructive response is allowed.

This is a standalone domain engine; an API/workflow may integrate it later.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from collections.abc import Mapping
from threading import RLock
import hashlib
import json
import math
import re

_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
_CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MAX_TRACKS = 512
_MAX_AMBIGUOUS = 512
_MAX_CANDIDATES = 512


@dataclass(frozen=True)
class AdvisoryPolicy:
    """Transparent evidence thresholds; never geographic risk thresholds."""
    min_observations: int = 3
    min_confidence: float = 0.85

    def __post_init__(self) -> None:
        if type(self.min_observations) is not int or not 2 <= self.min_observations <= 100:
            raise ValueError("min_observations must be an integer in [2,100]")
        if type(self.min_confidence) not in (int, float) or not math.isfinite(self.min_confidence) or not 0.5 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be a finite number in [0.5,1]")


def _valid_id(value: object, name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


def _int(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"invalid {name}")
    return value


def _confidence(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid track confidence")
    return float(value)


def _normalize(snapshot: Mapping, expected_session_id: str | None, expected_camera_id: str | None) -> dict:
    if not isinstance(snapshot, Mapping) or snapshot.get("state") != "running":
        raise ValueError("running session snapshot required")
    sid = _valid_id(snapshot.get("session_id"), "session_id", _SESSION_ID)
    if expected_session_id is not None and sid != expected_session_id:
        raise ValueError("session identity mismatch")
    cam = snapshot.get("camera_id")
    if cam is not None:
        cam = _valid_id(cam, "camera_id", _CAMERA_ID)
    if expected_camera_id is not None and cam != expected_camera_id:
        raise ValueError("camera identity mismatch")
    frame = _int(snapshot.get("frame_index"), "frame_index")
    raw_tracks = snapshot.get("tracks")
    raw_ambig = snapshot.get("ambiguous_tracks", [])
    if not isinstance(raw_tracks, list) or len(raw_tracks) > _MAX_TRACKS:
        raise ValueError("invalid or oversized tracks")
    if not isinstance(raw_ambig, list) or len(raw_ambig) > _MAX_AMBIGUOUS:
        raise ValueError("invalid or oversized ambiguous tracks")
    tracks: list[dict] = []
    ids: set[int] = set()
    for item in raw_tracks:
        if not isinstance(item, Mapping):
            raise ValueError("invalid track record")
        track_id = _int(item.get("track_id"), "track_id")
        if track_id in ids:
            raise ValueError("duplicate track identity")
        ids.add(track_id)
        label = item.get("class")
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,30}", label):
            raise ValueError("unsupported model class")
        state = item.get("state")
        if not isinstance(state, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,30}", state):
            raise ValueError("invalid tracker state")
        sample_frame = _int(item.get("frame_index"), "track frame_index")
        if sample_frame > frame:
            raise ValueError("future track observation")
        tracks.append({
            "track_id": track_id, "object_class": label,
            "tracker_state": state, "confidence": _confidence(item.get("confidence")),
            "frame_index": sample_frame,
        })
    tracks.sort(key=lambda t: t["track_id"])
    candidate_ids: set[int] = set()
    for item in raw_ambig:
        if not isinstance(item, Mapping) or item.get("state") != "ambiguous":
            raise ValueError("invalid ambiguous association")
        item_frame = _int(item.get("frame_index"), "ambiguity frame_index")
        if item_frame > frame:
            raise ValueError("future ambiguity frame")
        candidates = item.get("candidate_track_ids")
        if not isinstance(candidates, list) or len(candidates) > _MAX_CANDIDATES:
            raise ValueError("invalid ambiguity candidate IDs")
        for candidate in candidates:
            candidate_ids.add(_int(candidate, "ambiguity candidate track_id"))
    return {
        "session_id": sid, "camera_id": cam, "frame_index": frame,
        "tracks": tracks, "ambiguous_candidate_ids": sorted(candidate_ids),
        "ambiguous_count": len(raw_ambig),
    }


class EvidenceAlertEngine:
    """Thread-safe; holds state for one active session, no persistent archive.

    Same-frame replays are idempotent only if their normalized payload matches;
    old frames and altered same-frame data are rejected. Different track IDs
    are never merged by proximity or by image-path intersections.
    """

    def __init__(self, policy: AdvisoryPolicy | None = None):
        self.policy = policy or AdvisoryPolicy()
        self._lock = RLock()
        self._identity: tuple[str | None, str] | None = None
        self._last_frame: int = -1
        self._last_signature: str | None = None
        self._last_response: dict | None = None
        self._streaks: dict[int, int] = {}
        self._alerted: set[int] = set()

    def reset(self) -> None:
        with self._lock:
            self._identity = None
            self._last_frame = -1
            self._last_signature = None
            self._last_response = None
            self._streaks.clear()
            self._alerted.clear()

    def evaluate(
        self,
        snapshot: Mapping,
        *,
        expected_session_id: str | None = None,
        expected_camera_id: str | None = None,
    ) -> dict:
        record = _normalize(snapshot, expected_session_id, expected_camera_id)
        sig = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()
        identity = (record["camera_id"], record["session_id"])
        frame = record["frame_index"]
        with self._lock:
            if identity != self._identity:
                self._identity = identity
                self._last_frame = -1
                self._last_signature = None
                self._last_response = None
                self._streaks.clear()
                self._alerted.clear()
            if frame < self._last_frame:
                raise ValueError("out-of-order snapshot")
            if frame == self._last_frame:
                if sig != self._last_signature:
                    raise ValueError("conflicting snapshot for same frame")
                return deepcopy(self._last_response)

            ambig = set(record["ambiguous_candidate_ids"])
            next_streaks: dict[int, int] = {}
            next_alerted: set[int] = set()
            observations: list[dict] = []
            alerts: list[dict] = []
            for item in record["tracks"]:
                tid = item["track_id"]
                fresh = item["frame_index"] == frame
                eligible = (
                    record["camera_id"] is not None and fresh and
                    item["object_class"] == "drone" and
                    item["tracker_state"] == "confirmed" and
                    item["confidence"] >= self.policy.min_confidence and
                    tid not in ambig
                )
                streak = self._streaks.get(tid, 0) + 1 if eligible else 0
                if streak:
                    next_streaks[tid] = min(streak, self.policy.min_observations)
                ambiguous = tid in ambig
                evidence = (
                    "ambiguous_association" if ambiguous else
                    "eligible_model_evidence" if eligible else
                    "unregistered_camera" if record["camera_id"] is None else
                    "insufficient_or_stale_model_evidence"
                )
                observations.append({
                    "track_id": tid, "object_class": item["object_class"],
                    "tracker_state": item["tracker_state"],
                    "confidence": item["confidence"],
                    "model_evidence_status": evidence,
                    "successive_snapshot_observations": streak,
                    "physical_identity_verified": False,
                })
                if eligible and streak >= self.policy.min_observations:
                    next_alerted.add(tid)
                    if tid not in self._alerted:
                        alerts.append({
                            "event_type": "operator_review_advisory",
                            "severity": "attention",
                            "track_id": tid,
                            "object_class_model_label": "drone",
                            "confidence": item["confidence"],
                            "session_id": record["session_id"],
                            "frame_index": frame,
                            "camera_id": record["camera_id"],
                            "reason_codes": ["MODEL_DRONE_LABEL", "TRACKER_CONFIRMED_STATE", "REPEATED_FRAME_SNAPSHOTS"],
                            "ground_truth_verified": False,
                            "facility_breach_verified": False,
                            "target_gps": None,
                            "target_range_m": None,
                        })
                elif eligible and tid in self._alerted:
                    next_alerted.add(tid)
            self._streaks = next_streaks
            self._alerted = next_alerted
            result = {
                "session_id": record["session_id"],
                "camera_id": record["camera_id"],
                "frame_index": frame,
                "evaluation_basis": "model_and_tracker_evidence_only",
                "coordinate_system": "image_pixels",
                "target_gps_available": False,
                "facility_distance_available": False,
                "geofence_breach_verified": False,
                "automated_response_allowed": False,
                "ambiguous_association_count": record["ambiguous_count"],
                "observations": observations,
                "alerts": alerts,
                "policy": {
                    "min_observations": self.policy.min_observations,
                    "min_confidence": float(self.policy.min_confidence),
                },
            }
            self._last_frame = frame
            self._last_signature = sig
            self._last_response = deepcopy(result)
            return deepcopy(result)
