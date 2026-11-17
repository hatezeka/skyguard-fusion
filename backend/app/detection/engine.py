from dataclasses import dataclass
from typing import Literal
import math

import numpy as np


ObjectClass = Literal["drone", "bird", "other", "unknown"]


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def validate(self, image_width: int, image_height: int):
        values = (self.x, self.y, self.width, self.height)

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite bounding box")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid bounding box size")

        if self.x < 0 or self.y < 0:
            raise ValueError("Negative bounding box coordinates")

        if self.x + self.width > image_width:
            raise ValueError("Bounding box exceeds image width")

        if self.y + self.height > image_height:
            raise ValueError("Bounding box exceeds image height")


@dataclass(frozen=True)
class Detection:
    object_class: ObjectClass
    confidence: float
    bbox: BoundingBox

    def validate(self, image_width: int, image_height: int):
        if self.object_class not in {
            "drone", "bird", "other", "unknown"
        }:
            raise ValueError("Unsupported object class")

        if not math.isfinite(self.confidence):
            raise ValueError("Non-finite confidence")

        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence outside range")

        self.bbox.validate(image_width, image_height)


class DetectionEngine:
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError(
            "A real detection model must be connected"
        )


class DetectionValidator:
    @staticmethod
    def validate(
        image: np.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError("Invalid image")

        height, width = image.shape[:2]

        for detection in detections:
            detection.validate(width, height)

        return detections
