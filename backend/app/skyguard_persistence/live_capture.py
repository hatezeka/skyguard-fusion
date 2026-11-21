"""Live AI -> transactional Store bridge.

Only observes true detector outputs from the existing frame-processing worker.
No GPS or range is inferred, no UNKNOWN ID is assigned, no synthetic frames.
`observed_at` means host publication time, not sensor capture time.
"""
from __future__ import annotations

import hashlib
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from .database import Database
from .models import Camera, Session
from .schemas import CameraCreate, FrameIngest, ObjectInput, PerformanceCreate, SessionCreate
from .service import Store

log = logging.getLogger("skyguard.persistence.live")


@dataclass(frozen=True)
class ObservedFrame:
    live_session_id: str
    camera_ref: str
    source_kind: str
    frame_index: int
    observed_at: datetime
    processing_seconds: float | None
    image_sha256: str | None
    objects: tuple[ObjectInput, ...]


def _normalize_detection(d: Any, confirmed: dict[int, str]) -> ObjectInput:
    # A confirmed tracker observation must point to this very detection instance;
    # a close bounding box is NOT adequate scientific evidence of identity.
    cls = str(d.object_class).upper()
    if cls not in ("DRONE", "BIRD"):
        cls = "UNKNOWN"
    box = d.bbox
    xyxy = (float(box.x), float(box.y), float(box.x + box.width), float(box.y + box.height))
    track = confirmed.get(id(d)) if cls != "UNKNOWN" else None
    return ObjectInput(classification=cls, bbox=xyxy, confidence=float(d.confidence), track_id=track)


def capture_payload(snapshot: Any, detections: Any, tracks: Any, source_type: str,
                    source_location: object = None, jpeg_bytes: bytes | None = None,
                    clock=None) -> ObservedFrame:
    if getattr(snapshot, "state", None) != "running":
        raise ValueError("Only running frame snapshots can be archived")
    sid = getattr(snapshot, "session_id", None)
    frame = getattr(snapshot, "frame_index", None)
    if not isinstance(sid, str) or not sid or type(frame) is not int or frame < 0:
        raise ValueError("Live snapshot requires stable session ID and nonnegative frame index")
    if source_type not in ("usb", "rtsp", "file"):
        raise ValueError("Unknown original capture source")
    # Never store RTSP credentials or local file paths; fingerprint the location.
    locator_hash = hashlib.sha256(str(source_location).encode("utf8")).hexdigest()[:20]
    ident = getattr(snapshot, "camera_id", None)
    safe_identity = ident if isinstance(ident, str) and ident and len(ident) < 70 else "unregistered"
    camera_ref = f"live:{source_type}:{safe_identity}:{locator_hash}"
    matched = {}
    for t in tracks:
        if (getattr(t, "state", None) == "confirmed" and
                getattr(t, "frame_index", None) == frame and
                str(getattr(getattr(t, "detection", None), "object_class", "")).upper() in ("DRONE", "BIRD")):
            original_detection = getattr(t, "detection", None)
            key = id(original_detection)
            if key in matched:
                raise ValueError("A detection was assigned more than one confirmed identity")
            matched[key] = str(t.track_id)
    objects = tuple(_normalize_detection(d, matched) for d in detections)
    seconds = getattr(snapshot, "processing_seconds", None)
    if seconds is not None and (not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0):
        seconds = None
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    if now.tzinfo is None:
        raise ValueError("Host clock must be timezone aware")
    return ObservedFrame(
        live_session_id=sid, camera_ref=camera_ref, source_kind=source_type.upper(),
        frame_index=frame, observed_at=now.astimezone(timezone.utc),
        processing_seconds=seconds,
        image_sha256=hashlib.sha256(jpeg_bytes).hexdigest() if jpeg_bytes else None,
        objects=objects,
    )


class LiveRecorder:
    """Single writer, bounded queue, and explicit loss/error counters.

    Never silently samples a frame. A full queue results in an exposed loss counter.
    """

    def __init__(self, database: Database, *, queue_capacity: int = 2048):
        self.db = database
        self.store = Store(database)
        self.queue: queue.Queue[ObservedFrame] = queue.Queue(maxsize=queue_capacity)
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.manager = None
        self.lock = threading.Lock()
        self.ingested = 0
        self.duplicates = 0
        self.lost = 0
        self.errors = 0
        self.last_error: str | None = None
        self.last_observation: tuple[str, int, datetime] | None = None
        self.latest_event: ObservedFrame | None = None
        self._db_session_by_live: dict[str, str] = {}
        self._last_live: str | None = None
        self._closed: set[str] = set()
        self._last_frame_by_live: dict[str, int] = {}

    def start(self, manager: Any) -> None:
        if self.worker is not None:
            raise RuntimeError("Live bridge already started")
        self.manager = manager
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, name="skyguard-persistence-writer", daemon=True)
        self.worker.start()
        manager._skyguard_persistence_capture = self.capture

    def capture(self, *, snapshot, raw_detections, confirmed_tracks, source_type,
                source_location=None, jpeg_bytes=None) -> None:
        try:
            event = capture_payload(snapshot, raw_detections, confirmed_tracks, source_type, source_location, jpeg_bytes)
            self.queue.put_nowait(event)
            with self.lock:
                self.last_observation = (event.live_session_id, event.frame_index, event.observed_at)
                self.latest_event = event
        except queue.Full:
            self._failure("Persistence queue full; live frame NOT archived", lost=True)
        except (ValueError, TypeError, AttributeError) as exc:
            self._failure(f"Capture validation rejected frame: {exc}", lost=True)

    def _failure(self, reason: str, *, lost: bool) -> None:
        with self.lock:
            self.errors += 1
            if lost:
                self.lost += 1
            self.last_error = reason[:400]
        log.error(reason)

    def _resolve_session(self, event: ObservedFrame) -> str:
        known = self._db_session_by_live.get(event.live_session_id)
        if known:
            return known
        with self.db.sessionmaker() as tx:
            cam = tx.scalar(select(Camera).where(Camera.external_ref == event.camera_ref).limit(1))
        if cam is None:
            cam = self.store.create_camera(CameraCreate(
                name=f"SkyGuard live {event.source_kind} camera",
                source_kind=event.source_kind,
                external_ref=event.camera_ref,
            ), actor="live_bridge")
            camera_id = cam["id"]
        else:
            camera_id = cam.id
        ref = f"skyguard-live:{event.live_session_id}"
        with self.db.sessionmaker() as tx:
            existing = tx.scalar(select(Session).where(Session.dataset_ref == ref).limit(1))
        if existing:
            session_id = existing.id
        else:
            session_id = self.store.create_session(SessionCreate(
                camera_id=camera_id,
                source_kind="LIVE",
                validation_level="SOFTWARE_ONLY",
                dataset_ref=ref,
            ), actor="live_bridge")["id"]
        self._db_session_by_live[event.live_session_id] = session_id
        return session_id

    def _process(self, item: ObservedFrame) -> None:
        previous = self._last_frame_by_live.get(item.live_session_id)
        if previous is not None and item.frame_index <= previous:
            # A duplicate callback cannot create a second observation for one
            # frame. Out-of-order events must never be attached to new IDs.
            with self.lock:
                self.duplicates += 1
            return
        if self._last_live is not None and self._last_live != item.live_session_id:
            self._close(self._last_live)
        db_session = self._resolve_session(item)
        result = self.store.ingest_frame(FrameIngest(
            session_id=db_session,
            frame_index=item.frame_index,
            captured_at=item.observed_at,
            image_sha256=item.image_sha256,
            objects=list(item.objects),
        ), actor="live_bridge")
        if result["created"] and item.processing_seconds is not None:
            self.store.record_performance(PerformanceCreate(
                session_id=db_session,
                measured_at=item.observed_at,
                fps=None,  # Actual FPS cannot be established from one snapshot.
                latency_ms=item.processing_seconds * 1000,
                stage="live_backend_processing",
                method="perf_counter (detector + tracker + annotation + JPEG)",
            ), actor="live_bridge")
        with self.lock:
            if result["created"]:
                self.ingested += 1
            else:
                self.duplicates += 1
        self._last_live = item.live_session_id
        self._last_frame_by_live[item.live_session_id] = item.frame_index

    def _close(self, live_id: str) -> None:
        if live_id in self._closed:
            return
        db_id = self._db_session_by_live.get(live_id)
        if db_id:
            self.store.close_session(db_id, actor="live_bridge")
            self._closed.add(live_id)

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.15)
            except queue.Empty:
                self._maybe_close_finished()
                continue
            try:
                for attempt in range(3):
                    try:
                        self._process(event)
                        break
                    except Exception as exc:
                        if attempt == 2:
                            self._failure(f"DB write failed: {type(exc).__name__}: {exc}", lost=True)
                        else:
                            time.sleep(0.05 * (2 ** attempt))
            finally:
                self.queue.task_done()
        self._maybe_close_finished(force=True)

    def _maybe_close_finished(self, force=False):
        if not self._last_live or not self.queue.empty():
            return
        state = getattr(self.manager.status(), "state", "idle") if self.manager is not None else "idle"
        if force or state in ("stopped", "error", "idle"):
            try:
                self._close(self._last_live)
            except Exception as exc:
                self._failure(f"Session closure failed: {exc}", lost=False)

    def drain(self, timeout=10.0) -> bool:
        deadline = time.monotonic() + timeout
        while self.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return not self.queue.unfinished_tasks

    def stop(self, timeout=15.0) -> None:
        if self.manager is not None and getattr(self.manager, "_skyguard_persistence_capture", None) == self.capture:
            delattr(self.manager, "_skyguard_persistence_capture")
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout)
            if self.worker.is_alive():
                self._failure("Writer shutdown timeout; outstanding frames may remain in memory", lost=False)
            else:
                self.worker = None

    def health(self) -> dict:
        with self.lock:
            return {
                "status": "degraded" if self.lost or self.errors else "ok",
                "ingested_frames": self.ingested,
                "duplicate_frames": self.duplicates,
                "lost_frames": self.lost,
                "errors": self.errors,
                "last_error": self.last_error,
                "queued_frames": self.queue.qsize(),
                "capture_timestamp_semantics": "host_publication_utc_not_sensor_timestamp",
            }

    def observed_event(self, session_id: str | None, frame_index: int | None) -> ObservedFrame | None:
        with self.lock:
            event = self.latest_event
        if event is not None and event.live_session_id == session_id and event.frame_index == frame_index:
            return event
        return None

    def observation_time(self, session_id: str | None, frame_index: int | None) -> str | None:
        with self.lock:
            last = self.last_observation
        if last is not None and last[0] == session_id and last[1] == frame_index:
            return last[2].isoformat().replace("+00:00", "Z")
        return None
