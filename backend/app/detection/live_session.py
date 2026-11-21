"""Bounded, single-camera live detection session."""
import threading
from dataclasses import dataclass, field
from uuid import uuid4
from app.localization.session_calibration import freeze_session_calibration
from time import perf_counter
import cv2
from app.detection.engine import DetectionValidator
from app.detection.tracker import ObjectTracker
from app.video.source import UniversalVideoSource
from datetime import datetime, timezone
from app.detection.trajectory_history import TrajectoryHistory

@dataclass(frozen=True)
class SessionSnapshot:
    state: str
    frame_index: int | None
    detections: list[dict]
    tracks: list[dict]
    processing_seconds: float | None
    error: str | None
    camera_id: str | None = None
    ambiguous_tracks: list[dict] = field(default_factory=list)
    trajectories: list[dict] = field(default_factory=list)
    session_id: str | None = None

class LiveSessionManager:

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._snapshot = SessionSnapshot('idle', None, [], [], None, None)
        self._session_id: str | None = None
        self._bound_calibration = None
        self._jpeg = None

    def start(self, config, detector, camera_id=None, tracker=None, calibration_record=None):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError('A live session is already running')
            next_session_id = uuid4().hex
            next_calibration = freeze_session_calibration(
                calibration_record, camera_id, next_session_id
            )
            self._camera_id = camera_id
            self._session_id = next_session_id
            self._bound_calibration = next_calibration
            self._stop.clear()
            self._jpeg = None
            self._snapshot = SessionSnapshot('starting', None, [], [], None, None, camera_id=self._camera_id, session_id=self._session_id)
            self._thread = threading.Thread(target=self._worker, args=(config, detector, tracker), daemon=True, name='skyguard-live-camera')
            self._thread.start()

    def _worker(self, config, detector, tracker=None):
        tracker = tracker if tracker is not None else ObjectTracker()
        try:
            tracker.reset()
            trajectory_history = TrajectoryHistory()
            reset = getattr(detector, 'reset', None)
            if callable(reset):
                reset()
            with UniversalVideoSource(config) as source:
                while not self._stop.is_set():
                    frame = source.read()
                    if frame is None:
                        break
                    started = perf_counter()
                    detections = detector.detect(frame.image)
                    DetectionValidator.validate(frame.image, detections)
                    # Only target objects enter the production tracker; UNKNOWN remains a detection.
                    tracked_detections = [d for d in detections if d.object_class in ('drone', 'bird')]
                    tracking_result = tracker.update(
                        frame_index=frame.index,
                        detections=tracked_detections,
                    )
                    if (
                        hasattr(tracking_result, "confirmed")
                        and hasattr(tracking_result, "ambiguous")
                    ):
                        tracks = list(tracking_result.confirmed)
                        ambiguous = list(tracking_result.ambiguous)
                    else:
                        tracks = tracking_result
                        ambiguous = []
                    # Fail closed if an injected/custom tracker violates the target-only contract.
                    # Uncertain airborne objects stay in detections but never in tracks,
                    # candidate track IDs, trajectories, or drone alerts.
                    if any(
                        observation.detection.object_class not in ('drone', 'bird')
                        for observation in [*tracks, *ambiguous]
                    ):
                        raise ValueError('Non-target tracker output rejected: UNKNOWN/OTHER')
                    detection_data = [{'class': d.object_class, 'confidence': d.confidence, 'bbox': {'x': d.bbox.x, 'y': d.bbox.y, 'width': d.bbox.width, 'height': d.bbox.height}} for d in detections]
                    track_data = [
                        {
                            'track_id': t.track_id,
                            'class': t.detection.object_class,
                            'confidence': t.detection.confidence,
                            'state': t.state,
                            'frame_index': t.frame_index,
                        }
                        for t in tracks
                    ]
                    ambiguous_data = [
                        {
                            'class': t.detection.object_class,
                            'confidence': t.detection.confidence,
                            'state': 'ambiguous',
                            'frame_index': frame.index,
                            'candidate_track_ids': list(
                                t.candidate_track_ids
                            ),
                        }
                        for t in ambiguous
                    ]
                    confirmed_tracks = [
                        track for track in tracks
                        if getattr(track, "state", "confirmed")
                        == "confirmed"
                    ]

                    if confirmed_tracks:
                        trajectory_history.update(
                            confirmed_tracks,
                            datetime.now(timezone.utc),
                        )

                    trajectory_data = [
                        {
                            "track_id": track_id,
                            "object_class": points[-1].object_class,
                            "points": [
                                {
                                    "frame_index": point.frame_index,
                                    "timestamp": point.timestamp.isoformat(),
                                    "x_px": point.x_px,
                                    "y_px": point.y_px,
                                    "velocity_x_px_s": point.velocity_x_px_s,
                                    "velocity_y_px_s": point.velocity_y_px_s,
                                    "speed_px_s": point.speed_px_s,
                                    "confidence": point.confidence,
                                }
                                for point in points
                            ],
                        }
                        for track_id, points in sorted(
                            trajectory_history.get_all_histories().items()
                        )
                        if points
                    ]

                    annotated = frame.image.copy()
                    for t in tracks:
                        b = t.detection.bbox
                        p1 = (int(b.x), int(b.y))
                        p2 = (int(b.x + b.width), int(b.y + b.height))
                        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
                        cv2.putText(annotated, f'{t.detection.object_class} #{t.track_id}', (p1[0], max(15, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    for t in ambiguous:
                        b = t.detection.bbox
                        p1 = (int(b.x), int(b.y))
                        p2 = (
                            int(b.x + b.width),
                            int(b.y + b.height),
                        )
                        cv2.rectangle(
                            annotated, p1, p2, (0, 165, 255), 2
                        )
                        cv2.putText(
                            annotated,
                            f'{t.detection.object_class} ?',
                            (p1[0], max(15, p1[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 165, 255),
                            1,
                        )
                    ok, encoded = cv2.imencode(
                        '.jpg',
                        annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 75],
                    )
                    snapshot = SessionSnapshot(
                        state='running',
                        frame_index=frame.index,
                        detections=detection_data,
                        tracks=track_data,
                        processing_seconds=perf_counter() - started,
                        error=None,
                        camera_id=self._camera_id, session_id=self._session_id,
                        ambiguous_tracks=ambiguous_data,
                        trajectories=trajectory_data,
                    )
                    with self._lock:
                        if not self._stop.is_set():
                            self._snapshot = snapshot
                            self._jpeg = encoded.tobytes() if ok else None
                    # SkyGuard live persistence bridge: queue the original AI
                    # objects/confirmed tracker bindings, never reconstructed
                    # from browser polling or guessed bounding-box proximity.
                    if not self._stop.is_set():
                        callback = getattr(self, "_skyguard_persistence_capture", None)
                        if callback is not None:
                            callback(
                                snapshot=snapshot,
                                raw_detections=detections,
                                confirmed_tracks=tracks,
                                source_type=config.source_type,
                                source_location=config.location,
                                jpeg_bytes=encoded.tobytes() if ok else None,
                            )
            with self._lock:
                if self._snapshot.state != 'error':
                    self._snapshot = SessionSnapshot(
                        'stopped',
                        self._snapshot.frame_index,
                        self._snapshot.detections,
                        self._snapshot.tracks,
                        self._snapshot.processing_seconds,
                        None,
                        camera_id=self._camera_id, session_id=self._session_id,
                        ambiguous_tracks=self._snapshot.ambiguous_tracks,
                        trajectories=self._snapshot.trajectories,
                    )
        except Exception as exc:
            with self._lock:
                self._snapshot = SessionSnapshot(
                    'error',
                    self._snapshot.frame_index,
                    self._snapshot.detections,
                    self._snapshot.tracks,
                    self._snapshot.processing_seconds,
                    f'{type(exc).__name__}: {exc}',
                    camera_id=self._camera_id, session_id=self._session_id,
                    ambiguous_tracks=self._snapshot.ambiguous_tracks,
                    trajectories=self._snapshot.trajectories,
                )

    def stop(self, timeout=5.0):
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return self._snapshot
            self._stop.set()
            if self._snapshot.state in ('starting', 'running'):
                current = self._snapshot
                self._snapshot = SessionSnapshot(
                    'stopping',
                    current.frame_index,
                    current.detections,
                    current.tracks,
                    current.processing_seconds,
                    current.error,
                    camera_id=self._camera_id, session_id=self._session_id,
                    ambiguous_tracks=current.ambiguous_tracks,
                    trajectories=current.trajectories,
                )
            return self._snapshot

    def status(self):
        with self._lock:
            return self._snapshot

    def latest_frame(self):
        with self._lock:
            return self._snapshot, self._jpeg

    def latest_bound_frame(self):
        """One lock protects the image, detections, and frozen calibration."""
        with self._lock:
            return self._snapshot, self._jpeg, self._bound_calibration

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg
