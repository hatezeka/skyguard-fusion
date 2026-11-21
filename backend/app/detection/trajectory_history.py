"""Bounded per-object trajectory history in image coordinates."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from math import hypot, isfinite
from threading import RLock


@dataclass(frozen=True)
class TrajectoryPoint:
    track_id: int
    frame_index: int
    timestamp: datetime
    object_class: str
    confidence: float
    x_px: float
    y_px: float
    velocity_x_px_s: float | None
    velocity_y_px_s: float | None
    speed_px_s: float | None


class TrajectoryHistory:
    def __init__(
        self,
        max_points_per_track=300,
        max_tracks=1000,
    ):
        if (
            isinstance(max_points_per_track, bool)
            or not isinstance(max_points_per_track, int)
            or max_points_per_track < 2
        ):
            raise ValueError("Invalid max_points_per_track")

        if (
            isinstance(max_tracks, bool)
            or not isinstance(max_tracks, int)
            or max_tracks < 1
        ):
            raise ValueError("Invalid max_tracks")

        self.max_points_per_track = max_points_per_track
        self.max_tracks = max_tracks
        self._histories = {}
        self._lock = RLock()

    @staticmethod
    def _validate_timestamp(timestamp):
        if not isinstance(timestamp, datetime):
            raise ValueError("timestamp must be datetime")

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        return timestamp.astimezone(timezone.utc)

    def update(self, observations, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        timestamp = self._validate_timestamp(timestamp)
        observations = list(observations)

        prepared = []
        seen_ids = set()

        for observation in observations:
            track_id = observation.track_id
            frame_index = observation.frame_index
            detection = observation.detection
            bbox = detection.bbox

            if (
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id < 1
            ):
                raise ValueError("Invalid track_id")

            if track_id in seen_ids:
                raise ValueError("Duplicate track_id in one update")

            seen_ids.add(track_id)

            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise ValueError("Invalid frame_index")

            confidence = detection.confidence
            x = bbox.x + bbox.width / 2.0
            y = bbox.y + bbox.height / 2.0

            if not all(
                isfinite(value)
                for value in (confidence, x, y)
            ):
                raise ValueError("Non-finite observation")

            prepared.append(
                (
                    track_id,
                    frame_index,
                    detection.object_class,
                    float(confidence),
                    float(x),
                    float(y),
                )
            )

        with self._lock:
            new_ids = {
                item[0] for item in prepared
                if item[0] not in self._histories
            }

            if len(self._histories) + len(new_ids) > self.max_tracks:
                raise ValueError("Maximum track capacity exceeded")

            for track_id, frame, obj_class, confidence, x, y in prepared:
                history = self._histories.setdefault(
                    track_id,
                    deque(maxlen=self.max_points_per_track),
                )

                previous = history[-1] if history else None

                if previous is not None:
                    if frame <= previous.frame_index:
                        raise ValueError("Frame index must increase")

                    if timestamp <= previous.timestamp:
                        raise ValueError("Timestamp must increase")

                    elapsed = (timestamp - previous.timestamp).total_seconds()

                    vx = (x - previous.x_px) / elapsed
                    vy = (y - previous.y_px) / elapsed
                    speed = hypot(vx, vy)
                else:
                    vx = vy = speed = None

                point = TrajectoryPoint(
                    track_id=track_id,
                    frame_index=frame,
                    timestamp=timestamp,
                    object_class=obj_class,
                    confidence=confidence,
                    x_px=x,
                    y_px=y,
                    velocity_x_px_s=vx,
                    velocity_y_px_s=vy,
                    speed_px_s=speed,
                )

                history.append(point)

        return tuple(
            point
            for track_id, *_ in prepared
            for point in (self.get_history(track_id)[-1],)
        )

    def get_history(self, track_id):
        with self._lock:
            return tuple(self._histories.get(track_id, ()))

    def get_all_histories(self):
        with self._lock:
            return {
                track_id: tuple(history)
                for track_id, history in self._histories.items()
            }

    def remove_track(self, track_id):
        with self._lock:
            return self._histories.pop(track_id, None) is not None

    def reset(self):
        with self._lock:
            self._histories.clear()

    @property
    def track_count(self):
        with self._lock:
            return len(self._histories)
