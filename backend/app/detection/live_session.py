"""Bounded live detection session using the primary camera tracker."""
import base64
import os
import threading
from dataclasses import asdict, dataclass, field
from uuid import uuid4
from app.localization.session_calibration import freeze_session_calibration
from time import perf_counter
import cv2
from app.detection.engine import DetectionValidator
from app.detection.tracker import ObjectTracker
from app.video.source import UniversalVideoSource
from app.video.stereo_capture import StereoCapture
from app.detection.stereo_observation import load_stereo_profile, stereo_observations
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
    stereo: dict | None = None

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
            right_index = getattr(config, 'stereo_right_index', None)
            stereo = right_index is not None
            profile = load_stereo_profile(os.environ.get("SKYGUARD_STEREO_PROFILE")) if stereo else None
            capture = StereoCapture(config.location, right_index) if stereo else UniversalVideoSource(config)
            with capture as source:
                while not self._stop.is_set():
                    if stereo:
                        left, right = source.next_pair(cancel=self._stop)
                        frame = left.frame
                    else:
                        frame = source.read()
                    if frame is None:
                        break
                    started = perf_counter()
                    detections = detector.detect(frame.image)
                    left_inference_ms = (perf_counter() - started) * 1000
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
                    ids = {id(track.detection): track.track_id for track in tracks}
                    colors = {'drone': (0, 230, 0), 'bird': (255, 200, 0), 'unknown': (0, 160, 255)}
                    for detection in detections:
                        label = detection.object_class if detection.object_class in colors else 'unknown'
                        b = detection.bbox
                        p1 = (int(b.x), int(b.y))
                        p2 = (int(b.x + b.width), int(b.y + b.height))
                        color = colors[label]
                        tid = ids.get(id(detection))
                        identity = f' #{tid}' if tid is not None and label != 'unknown' else ''
                        confidence = '' if label == 'unknown' else f' {detection.confidence:.2f}'
                        cv2.rectangle(annotated, p1, p2, color, 2)
                        cv2.putText(annotated, f'{label.upper()}{identity}{confidence}',
                                    (p1[0], max(15, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
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
                    stereo_data = None
                    if stereo:
                        right_started = perf_counter()
                        right_detections = detector.detect(right.frame.image)
                        right_inference_ms = (perf_counter() - right_started) * 1000
                        DetectionValidator.validate(right.frame.image, right_detections)
                        observations = stereo_observations(frame.image, right.frame.image, tracks, right_detections, profile)
                        right_render_started = perf_counter()
                        right_image = right.frame.image.copy()
                        right_ids = {item.right_bbox: item.track_id for item in observations if item.right_bbox}
                        for detection in right_detections:
                            box = detection.bbox
                            label = detection.object_class if detection.object_class in colors else 'unknown'
                            track_id = right_ids.get((box.x, box.y, box.width, box.height))
                            identity = f' #{track_id}' if label != 'unknown' and track_id is not None else ''
                            x, y = int(box.x), int(box.y)
                            cv2.rectangle(right_image, (x, y), (int(box.x+box.width), int(box.y+box.height)), colors[label], 2)
                            cv2.putText(right_image, f'{label.upper()}{identity} {detection.confidence:.2f}',
                                        (x, max(15,y-8)), cv2.FONT_HERSHEY_SIMPLEX, .5, colors[label], 1)
                        right_ok, right_jpeg = cv2.imencode('.jpg', right_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        stereo_data = {
                            "right_image_base64": base64.b64encode(right_jpeg.tobytes()).decode("ascii") if right_ok else None,
                            "left_inference_ms": left_inference_ms,
                            "right_inference_ms": right_inference_ms,
                            "right_render_ms": (perf_counter() - right_render_started) * 1000,
                            "profile_sha256": profile.sha256 if profile else None,
                            "calibration_status": "CALIBRATION_SUSPECT" if profile else "CALIBRATION_UNAVAILABLE",
                            "timestamp_semantics": "HOST_READ_RETURN_NOT_EXPOSURE",
                            "right_frame_index": right.frame.index,
                            "host_skew_ms": abs(left.timestamp_ns - right.timestamp_ns) / 1e6,
                            "range_status": "RANGE_UNAVAILABLE",
                            "objects": [asdict(item) for item in observations],
                            "cameras": source.report(),
                        }
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
                        stereo=stereo_data,
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
                    'stopped' if self._stop.is_set() else 'error',
                    self._snapshot.frame_index,
                    self._snapshot.detections,
                    self._snapshot.tracks,
                    self._snapshot.processing_seconds,
                    ('USB stereo: '+str(exc) if getattr(config, 'stereo_right_index', None) is not None and not self._stop.is_set() else 'RTSP stream unavailable' if getattr(config, 'source_type', None) == 'rtsp'
                     else f'{type(exc).__name__}: {exc}'),
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
