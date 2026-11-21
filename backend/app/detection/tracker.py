from dataclasses import dataclass
import math

from app.detection.engine import BoundingBox, Detection


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x + first.width, second.x + second.width)
    y2 = min(first.y + first.height, second.y + second.height)

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = first.width * first.height
    second_area = second.width * second.height
    union = first_area + second_area - intersection

    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    detection: Detection
    frame_index: int
    hits: int
    state: str


@dataclass
class _Track:
    track_id: int
    detection: Detection
    last_frame: int
    hits: int = 1


class ObjectTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missing_frames: int = 5,
        min_confirmed_hits: int = 3,
    ):
        if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError("Invalid IoU threshold")
        if max_missing_frames < 0:
            raise ValueError("Invalid max_missing_frames")
        if min_confirmed_hits < 1:
            raise ValueError("Invalid min_confirmed_hits")

        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.min_confirmed_hits = min_confirmed_hits
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_frame: int | None = None

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._last_frame = None

    def update(
        self,
        frame_index: int,
        detections: list[Detection],
    ) -> list[TrackObservation]:
        if frame_index < 0:
            raise ValueError("Negative frame index")

        if self._last_frame is not None and frame_index <= self._last_frame:
            raise ValueError("Frame indices must increase")

        for detection in detections:
            detection.validate(
                image_width=10**9,
                image_height=10**9,
            )

        self._last_frame = frame_index

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame - 1 > self.max_missing_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

        candidates = []

        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = box_iou(track.detection.bbox, detection.bbox)
                if overlap >= self.iou_threshold:
                    candidates.append(
                        (-overlap, track_id, detection_index)
                    )

        candidates.sort()
        assigned_tracks = set()
        assigned_detections = set()
        results = []

        for _, track_id, detection_index in candidates:
            if track_id in assigned_tracks:
                continue
            if detection_index in assigned_detections:
                continue

            track = self._tracks[track_id]
            detection = detections[detection_index]

            track.detection = detection
            track.last_frame = frame_index
            track.hits += 1

            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

            results.append(
                self._observation(track, frame_index)
            )

        for index, detection in enumerate(detections):
            if index in assigned_detections:
                continue

            track = _Track(
                track_id=self._next_id,
                detection=detection,
                last_frame=frame_index,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1

            results.append(
                self._observation(track, frame_index)
            )

        return sorted(results, key=lambda item: item.track_id)

    def _observation(
        self,
        track: _Track,
        frame_index: int,
    ) -> TrackObservation:
        state = (
            "confirmed"
            if track.hits >= self.min_confirmed_hits
            else "tentative"
        )

        return TrackObservation(
            track_id=track.track_id,
            detection=track.detection,
            frame_index=frame_index,
            hits=track.hits,
            state=state,
        )