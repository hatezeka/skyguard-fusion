from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from app.detection.engine import Detection


@dataclass(frozen=True)
class ByteTrackObservation:
    track_id: int
    detection_index: int
    detection: Detection


class _ByteTrackDetections:
    def __init__(
        self,
        xywh: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return _ByteTrackDetections(
            self.xywh[index],
            self.conf[index],
            self.cls[index],
        )


class ByteTrackAdapter:
    def __init__(
        self,
        frame_rate: int = 30,
        track_buffer: int = 30,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.25,
        match_thresh: float = 0.80,
    ):
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        if track_buffer < 0:
            raise ValueError("track_buffer cannot be negative")

        for name, value in (
            ("track_high_thresh", track_high_thresh),
            ("track_low_thresh", track_low_thresh),
            ("new_track_thresh", new_track_thresh),
            ("match_thresh", match_thresh),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name}")

        if track_low_thresh > track_high_thresh:
            raise ValueError(
                "track_low_thresh cannot exceed track_high_thresh"
            )

        self._config = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )

        self._frame_rate = frame_rate
        self._tracker = self._new_tracker()
        self._last_frame = None

    def _new_tracker(self):
        tracker = BYTETracker(args=self._config)
        tracker.frame_rate = self._frame_rate
        return tracker

    def reset(self):
        self._tracker = self._new_tracker()
        self._last_frame = None

    def update(
        self,
        frame_index: int,
        detections: list[Detection],
    ) -> list[ByteTrackObservation]:
        if frame_index < 0:
            raise ValueError("frame_index cannot be negative")

        if (
            self._last_frame is not None
            and frame_index <= self._last_frame
        ):
            raise ValueError(
                "frame_index must increase strictly"
            )

        if (
            self._last_frame is not None
            and frame_index != self._last_frame + 1
        ):
            raise ValueError(
                "ByteTrack requires consecutive frame indices"
            )

        xywh = []
        conf = []
        cls = []

        class_mapping = {
            "drone": 0,
            "bird": 1,
            "other": 2,
            "unknown": 3,
        }

        for detection in detections:
            detection.validate(
                image_width=10**9,
                image_height=10**9,
            )

            box = detection.bbox

            xywh.append([
                box.x + box.width / 2,
                box.y + box.height / 2,
                box.width,
                box.height,
            ])

            conf.append(detection.confidence)
            cls.append(
                class_mapping.get(detection.object_class, 2)
            )

        results = _ByteTrackDetections(
            xywh=np.asarray(xywh, dtype=np.float32).reshape(-1, 4),
            conf=np.asarray(conf, dtype=np.float32),
            cls=np.asarray(cls, dtype=np.float32),
        )

        tracked = self._tracker.update(results)
        self._last_frame = frame_index

        observations = []

        for row in tracked:
            detection_index = int(row[7])

            if not 0 <= detection_index < len(detections):
                raise ValueError(
                    "ByteTrack returned invalid detection index"
                )

            observations.append(
                ByteTrackObservation(
                    track_id=int(row[4]),
                    detection_index=detection_index,
                    detection=detections[detection_index],
                )
            )

        return observations
