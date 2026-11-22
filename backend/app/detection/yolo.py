from pathlib import Path
import math

import numpy as np
from skyguard_fix.core import Detection as ObjectProposal, merge_object_proposals

from app.detection.engine import (
    BoundingBox,
    Detection,
    DetectionEngine,
    DetectionValidator,
)


class YOLODetectionEngine(DetectionEngine):
    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float = 0.25,
        model=None,
    ):
        if not math.isfinite(confidence_threshold) or not (
            0 <= confidence_threshold <= 1
        ):
            raise ValueError("Invalid confidence threshold")

        self.consolidate_proposals = model is None
        if model is None:
            if not model_path:
                raise ValueError("Model path is required")

            path = Path(model_path)

            if not path.is_file():
                raise FileNotFoundError(path)

            from ultralytics import YOLO
            model = YOLO(str(path))
            if model.task != "detect":
                raise ValueError("Checkpoint must be an object detector")
            labels = {index: self.map_class(name) for index, name in model.names.items()}
            expected = {0: "drone", 1: "bird"}
            if labels not in (expected, {**expected, 2: "unknown"}):
                raise ValueError("Unexpected checkpoint classes")

        self.model = model
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def map_class(name: str) -> str:
        normalized = name.strip().lower()

        if normalized in {"drone", "uav", "quadcopter"}:
            return "drone"

        if normalized == "bird":
            return "bird"

        if normalized in {"unknown", "unidentified"}:
            return "unknown"

        return "other"

    def detect(self, image: np.ndarray) -> list[Detection]:
        DetectionValidator.validate(image, [])

        results = self.model.predict(
            source=image,
            conf=max(self.confidence_threshold, 0.001),
            verbose=False,
            device="cpu",
        )

        if len(results) != 1:
            raise ValueError("Expected exactly one inference result")

        result = results[0]
        height, width = image.shape[:2]
        detections = []

        if result.boxes is None:
            return detections

        names = result.names

        for box in result.boxes:
            confidence = float(box.conf.item())

            if not math.isfinite(confidence):
                raise ValueError("Non-finite model confidence")

            if confidence < self.confidence_threshold:
                continue

            class_id = int(box.cls.item())

            if isinstance(names, dict):
                class_name = names.get(class_id)
            else:
                class_name = (
                    names[class_id]
                    if 0 <= class_id < len(names)
                    else None
                )

            if not isinstance(class_name, str):
                raise ValueError("Unknown model class ID")

            coordinates = box.xyxy[0].tolist()

            if len(coordinates) != 4:
                raise ValueError("Invalid model bounding box")

            x1, y1, x2, y2 = map(float, coordinates)

            if not all(
                math.isfinite(v)
                for v in (x1, y1, x2, y2)
            ):
                raise ValueError("Non-finite model coordinates")

            x1 = max(0.0, min(float(width), x1))
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    object_class=self.map_class(class_name),
                    confidence=confidence,
                    bbox=BoundingBox(
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                    ),
                )
            )

        if self.consolidate_proposals:
            proposals = [
                ObjectProposal(
                    d.object_class.upper(), d.confidence,
                    (d.bbox.x, d.bbox.y, d.bbox.x + d.bbox.width, d.bbox.y + d.bbox.height),
                    'live_primary',
                )
                for d in detections if d.object_class in ('drone', 'bird', 'unknown')
            ]
            finalized = [d for d in detections if d.object_class == 'other']
            for proposal in merge_object_proposals(proposals):
                x1, y1, x2, y2 = proposal.xyxy
                label = 'unknown' if proposal.verdict_status == 'CONFLICT_REVIEW' else self.map_class(proposal.label)
                finalized.append(Detection(label, proposal.confidence, BoundingBox(x1, y1, x2 - x1, y2 - y1)))
            detections = finalized
        return DetectionValidator.validate(image, detections)
