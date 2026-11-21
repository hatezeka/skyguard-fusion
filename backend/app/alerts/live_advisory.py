"""Phase 41.2: bounded, process-local bridge from live snapshots to advisories.

This is an on-demand evaluation API, NOT a background frame processor. Polling
may skip camera frames; its evidence count means successive *evaluated* frames.
There is no persistent alert queue, verified target location, or auto-response.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from threading import RLock
from typing import Any

from app.alerts.risk_engine import EvidenceAlertEngine


def _snapshot_dict(snapshot: object) -> dict[str, Any]:
    if is_dataclass(snapshot) and not isinstance(snapshot, type):
        return asdict(snapshot)
    if isinstance(snapshot, Mapping):
        return deepcopy(dict(snapshot))
    raise ValueError("Live snapshot must be a dataclass or mapping")


class LiveAdvisoryBridge:
    """Serializes per-process request evaluations and checks active identity.

    All event IDs are stable under same-frame replays. Different camera sessions
    cannot use the same event ID. This class does not claim exactly-once network
    delivery or track continuity across independently observed camera frames.
    """

    def __init__(self, engine: EvidenceAlertEngine | None = None) -> None:
        self._lock = RLock()
        self._engine = engine or EvidenceAlertEngine()

    def reset(self) -> None:
        with self._lock:
            self._engine.reset()

    def evaluate(self, read_status: Callable[[], object]) -> dict[str, Any]:
        with self._lock:
            before = _snapshot_dict(read_status())
            if before.get("state") != "running":
                self._engine.reset()
                raise ValueError("No running camera session")
            # The source is server-controlled; callers cannot submit track IDs,
            # guessed frames, class labels or arbitrary evidence for evaluation.
            evaluated = self._engine.evaluate(
                before,
                expected_session_id=before.get("session_id"),
                expected_camera_id=before.get("camera_id"),
            )
            after = _snapshot_dict(read_status())
            if (
                after.get("state") != "running"
                or after.get("session_id") != evaluated["session_id"]
                or after.get("camera_id") != evaluated["camera_id"]
                or type(after.get("frame_index")) is not int
                or after["frame_index"] < evaluated["frame_index"]
            ):
                self._engine.reset()
                raise ValueError("Live session changed during advisory evaluation")

            # API transport idempotency: re-reading the same frame gives the
            # SAME event identity. It must not be counted as a new observation.
            for alert in evaluated["alerts"]:
                seed = (
                    f'{evaluated["camera_id"]}:{evaluated["session_id"]}:'
                    f'{evaluated["frame_index"]}:{alert["track_id"]}:'
                    f'{alert["event_type"]}'
                )
                alert["event_id"] = sha256(seed.encode("utf-8")).hexdigest()

            evaluated["sampling_mode"] = "on_demand_latest_snapshot"
            evaluated["sampled_frames_only"] = True
            evaluated["latest_frame_index_at_check"] = after["frame_index"]
            evaluated["alert_archive_persistent"] = False
            evaluated["ground_truth_verified"] = False
            return evaluated
