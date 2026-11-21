"""Motion-aware multi-object tracking with explicit identity ambiguity."""

from dataclasses import dataclass
from math import hypot, isfinite

from app.detection.engine import Detection
from app.detection.motion_aware_tracker import _hungarian


@dataclass(frozen=True)
class ConfirmedObservation:
    track_id: int
    detection_index: int
    detection: Detection


@dataclass(frozen=True)
class AmbiguousObservation:
    detection_index: int
    detection: Detection
    candidate_track_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrackingResult:
    confirmed: list[ConfirmedObservation]
    ambiguous: list[AmbiguousObservation]


@dataclass
class _Track:
    track_id: int
    object_class: str
    x: float
    y: float
    vx: float
    vy: float
    last_frame: int


class MotionAwareTrackerV2:
    def __init__(
        self,
        max_distance=50.0,
        max_missing_frames=5,
        ambiguity_margin=0.01,
    ):
        if not isfinite(max_distance) or max_distance <= 0:
            raise ValueError("Invalid max_distance")

        if (
            isinstance(max_missing_frames, bool)
            or not isinstance(max_missing_frames, int)
            or max_missing_frames < 0
        ):
            raise ValueError("Invalid max_missing_frames")

        if not isfinite(ambiguity_margin) or ambiguity_margin < 0:
            raise ValueError("Invalid ambiguity_margin")

        self.max_distance = float(max_distance)
        self.max_missing_frames = max_missing_frames
        self.ambiguity_margin = float(ambiguity_margin)
        self.reset()

    def reset(self):
        self._tracks = {}
        self._next_id = 1
        self._last_frame = None

    @staticmethod
    def _center(detection):
        box = detection.bbox
        return (
            box.x + box.width / 2.0,
            box.y + box.height / 2.0,
        )

    def update(self, frame_index: int, detections: list[Detection]):
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError("Invalid frame index")

        if self._last_frame is not None:
            if frame_index != self._last_frame + 1:
                raise ValueError("Frames must be consecutive")

        for detection in detections:
            detection.validate(
                image_width=10**9,
                image_height=10**9,
            )

        centers = [self._center(d) for d in detections]

        tracks = sorted(
            self._tracks.values(),
            key=lambda t: t.track_id,
        )

        predictions = {}

        for track in tracks:
            dt = frame_index - track.last_frame
            predictions[track.track_id] = (
                track.x + track.vx * dt,
                track.y + track.vy * dt,
            )

        # Detect near-equal association costs before Hungarian matching.
        ambiguous = []
        ambiguous_indices = set()
        reserved_track_ids = set()

        for j, detection in enumerate(detections):
            cx, cy = centers[j]
            candidates = []

            for track in tracks:
                if not (track.object_class == detection.object_class or 'unknown' in (track.object_class, detection.object_class)):
                    continue

                px, py = predictions[track.track_id]
                distance = hypot(cx - px, cy - py)

                if distance <= self.max_distance:
                    candidates.append((distance, track.track_id))

            candidates.sort()

            if len(candidates) < 2:
                continue

            best_distance = candidates[0][0]

            close_ids = tuple(
                track_id
                for distance, track_id in candidates
                if distance - best_distance <= self.ambiguity_margin
            )

            if len(close_ids) >= 2:
                ambiguous_indices.add(j)
                reserved_track_ids.update(close_ids)

                ambiguous.append(
                    AmbiguousObservation(
                        detection_index=j,
                        detection=detection,
                        candidate_track_ids=tuple(sorted(close_ids)),
                    )
                )

        # Exclude ambiguous detections and their candidate tracks.
        # They retain their previous motion state for later recovery.
        available_tracks = [
            track for track in tracks
            if track.track_id not in reserved_track_ids
        ]

        available_indices = [
            j for j in range(len(detections))
            if j not in ambiguous_indices
        ]

        n = len(available_tracks)
        penalty = self.max_distance + 1.0
        invalid = penalty * 1000000.0

        costs = []

        for track in available_tracks:
            px, py = predictions[track.track_id]
            row = []

            for j in available_indices:
                detection = detections[j]
                cx, cy = centers[j]

                distance = hypot(cx - px, cy - py)

                if (
                    not (detection.object_class == track.object_class or 'unknown' in (detection.object_class, track.object_class))
                    or distance > self.max_distance
                ):
                    row.append(invalid)
                else:
                    row.append(distance)

            row.extend([penalty] * n)
            costs.append(row)

        assignments = _hungarian(costs)

        confirmed = []
        matched_indices = set()

        for i, track in enumerate(available_tracks):
            column = assignments[i] if assignments else -1

            if not 0 <= column < len(available_indices):
                continue

            if costs[i][column] >= penalty:
                continue

            j = available_indices[column]
            cx, cy = centers[j]
            dt = frame_index - track.last_frame

            track.vx = (cx - track.x) / dt
            track.vy = (cy - track.y) / dt
            track.x = cx
            track.y = cy
            track.last_frame = frame_index
            track.object_class = detections[j].object_class

            matched_indices.add(j)

            confirmed.append(
                ConfirmedObservation(
                    track_id=track.track_id,
                    detection_index=j,
                    detection=detections[j],
                )
            )

        # Expire tracks only after the allowed missing interval.
        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame > self.max_missing_frames
        ]

        for track_id in expired_ids:
            del self._tracks[track_id]

        # Unmatched, non-ambiguous detections become new tracks.
        for j, detection in enumerate(detections):
            if j in matched_indices or j in ambiguous_indices:
                continue

            cx, cy = centers[j]
            track_id = self._next_id
            self._next_id += 1

            self._tracks[track_id] = _Track(
                track_id=track_id,
                object_class=detection.object_class,
                x=cx,
                y=cy,
                vx=0.0,
                vy=0.0,
                last_frame=frame_index,
            )

            confirmed.append(
                ConfirmedObservation(
                    track_id=track_id,
                    detection_index=j,
                    detection=detection,
                )
            )

        self._last_frame = frame_index

        return TrackingResult(
            confirmed=sorted(
                confirmed,
                key=lambda o: o.track_id,
            ),
            ambiguous=sorted(
                ambiguous,
                key=lambda o: o.detection_index,
            ),
        )
