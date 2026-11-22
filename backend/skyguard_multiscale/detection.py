"""Auditable coarse-to-fine four-class YOLO inference.

Large = sqrt(box area / frame area) >= 0.10
Medium = >= 0.035 and < 0.10
Small = < 0.035

Stages run in order: full frame at 640, full frame at 1280, overlapping
640-pixel tiles. Each stage contributes only its target size tier.
Coordinates always refer to the ORIGINAL unresized frame.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Any


TARGET_CLASSES = ('DRONE', 'BIRD')
ALL_CLASSES = ('DRONE', 'BIRD', 'AIRPLANE', 'HELICOPTER')


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    tier: str
    stage: str
    track_id: int | None = None

    def record(self) -> dict:
        return asdict(self)


def validate_box(box, width: int, height: int) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError('BBOX_NOT_XYXY')
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(map(math.isfinite, (x1, y1, x2, y2))):
        raise ValueError('NONFINITE_BOX')
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('INVALID_BOX')
    return (x1, y1, x2, y2)


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    union = aa + bb - intersection
    return intersection / union if union > 0 else 0.0


def tier_for(box, width: int, height: int, large_min=0.10, medium_min=0.035):
    if not 0 < medium_min < large_min < 1:
        raise ValueError('INVALID_TIER_THRESHOLDS')
    fraction = ((box[2]-box[0])*(box[3]-box[1])/(width*height))**0.5
    return 'large' if fraction >= large_min else 'medium' if fraction >= medium_min else 'small'


def tile_windows(width: int, height: int, size: int = 640, overlap: float = .25):
    if not (0 <= overlap < .8) or size < 32 or width < 1 or height < 1:
        raise ValueError('INVALID_TILE_SPEC')
    def origins(length):
        if length <= size:
            return [0]
        step = max(1, int(size*(1-overlap)))
        return sorted(set(list(range(0, length-size+1, step)) + [length-size]))
    return [(x, y, min(x+size, width), min(y+size, height))
            for y in origins(height) for x in origins(width)]


def suppress(detections: list[Detection], threshold=.45):
    """Class-aware NMS; UNKNOWN candidates may suppress each other only."""
    result = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if any(other.class_name == det.class_name and
               box_iou(other.xyxy, det.xyxy) >= threshold for other in result):
            continue
        result.append(det)
    return result


class YOLOAdapter:
    """Actual inference; requires an explicitly supplied trained checkpoint."""
    def __init__(self, weights: str):
        from pathlib import Path
        if not Path(weights).is_file():
            raise FileNotFoundError(f'TRAINED_CHECKPOINT_MISSING: {weights}')
        from ultralytics import YOLO
        self.model = YOLO(weights)
        names = self.model.names
        self.names = {int(k): str(v).upper() for k,v in
                      (names.items() if isinstance(names, dict) else enumerate(names))}
        if set(self.names.values()) != set(ALL_CLASSES) or len(self.names) != 4:
            raise ValueError(f'CHECKPOINT_CLASS_CONTRACT_FAILED: {self.names}')

    def predict(self, image, imgsz: int, confidence: float):
        results = self.model.predict(source=image, imgsz=imgsz, conf=confidence,
                                     verbose=False, save=False)
        out = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = self.names[int(box.cls.item())]
                out.append((cls, float(box.conf.item()),
                            tuple(float(v) for v in box.xyxy[0].tolist())))
        return out


class CoarseToFine:
    def __init__(self, model: Any, *, min_confidence=.25, accept_confidence=.60,
                 tile_size=640, tile_overlap=.25, max_tiles=128, nms_iou=.45):
        if not 0 < min_confidence <= accept_confidence <= 1:
            raise ValueError('INVALID_CONFIDENCE_THRESHOLDS')
        self.model = model
        self.min_confidence, self.accept_confidence = min_confidence, accept_confidence
        self.tile_size, self.tile_overlap, self.max_tiles = tile_size, tile_overlap, max_tiles
        self.nms_iou = nms_iou

    def detect(self, image):
        height, width = image.shape[:2]
        if width < 1 or height < 1:
            raise ValueError('EMPTY_FRAME')
        windows = tile_windows(width, height, self.tile_size, self.tile_overlap)
        if len(windows) > self.max_tiles:
            raise ValueError(f'TILE_COVERAGE_EXCEEDS_LIMIT: {len(windows)} > {self.max_tiles}')
        results = []
        def collect(sub_image, offset_x, offset_y, stage, target, imgsz):
            for class_name, conf, local_box in self.model.predict(sub_image, imgsz, self.min_confidence):
                class_name = str(class_name).upper()
                if class_name not in ALL_CLASSES:
                    raise ValueError(f'UNEXPECTED_MODEL_CLASS: {class_name}')
                try:
                    global_box = validate_box((local_box[0]+offset_x, local_box[1]+offset_y,
                                               local_box[2]+offset_x, local_box[3]+offset_y),
                                              width, height)
                except ValueError:
                    continue
                tier = tier_for(global_box, width, height)
                if tier != target:
                    continue
                # Airplane, helicopter and uncertain target-class predictions
                # are rejected as UNKNOWN, without tracks.
                final_class = class_name if class_name in TARGET_CLASSES and conf >= self.accept_confidence else 'UNKNOWN'
                results.append(Detection(final_class, float(conf), global_box, tier, stage))

        collect(image, 0, 0, 'full_640', 'large', 640)
        collect(image, 0, 0, 'full_1280', 'medium', 1280)
        for x1, y1, x2, y2 in windows:
            collect(image[y1:y2, x1:x2], x1, y1, 'tiles_640', 'small', 640)
        return suppress(results, self.nms_iou)


class IoUTracker:
    """Simple transparent baseline tracker, NOT a claim of IDF1/HOTA quality."""
    def __init__(self, min_iou=.30, max_gap=5):
        self.min_iou, self.max_gap = min_iou, max_gap
        self.tracks = {}
        self.next_id = 1
        self.last_frame = -1

    def update(self, frame_index: int, detections: list[Detection]):
        if frame_index <= self.last_frame:
            raise ValueError('NONMONOTONIC_FRAME_INDEX')
        self.last_frame = frame_index
        self.tracks = {tid: value for tid,value in self.tracks.items()
                       if frame_index-value[2] <= self.max_gap+1}
        used = set()
        output = []
        for det in sorted(detections, key=lambda d:d.confidence, reverse=True):
            if det.class_name not in TARGET_CLASSES:
                output.append(det)  # UNKNOWN has no ID and no trajectory
                continue
            options = sorted(((box_iou(det.xyxy, last_box), tid)
                              for tid,(last_cls,last_box,_) in self.tracks.items()
                              if tid not in used and last_cls==det.class_name), reverse=True)
            score, tid = options[0] if options else (0.0, None)
            if tid is None or score < self.min_iou:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = (det.class_name, det.xyxy, frame_index)
            used.add(tid)
            output.append(Detection(det.class_name, det.confidence, det.xyxy,
                                    det.tier, det.stage, tid))
        return output
