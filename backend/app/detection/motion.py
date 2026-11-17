import cv2
import numpy as np

from app.detection.engine import (
    BoundingBox,
    Detection,
    DetectionEngine,
    DetectionValidator,
)


class MotionDetectionEngine(DetectionEngine):
    def __init__(
        self,
        min_area: int = 25,
        threshold: int = 20,
    ):
        if min_area <= 0:
            raise ValueError("min_area must be positive")

        if not 0 < threshold < 256:
            raise ValueError("threshold must be between 1 and 255")

        self.min_area = min_area
        self.threshold = threshold
        self.previous_gray = None

    def reset(self):
        self.previous_gray = None

    def detect(self, image: np.ndarray) -> list[Detection]:
        DetectionValidator.validate(image, [])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return []

        difference = cv2.absdiff(self.previous_gray, gray)

        _, mask = cv2.threshold(
            difference,
            self.threshold,
            255,
            cv2.THRESH_BINARY,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            detections.append(
                Detection(
                    object_class="unknown",
                    confidence=0.5,
                    bbox=BoundingBox(
                        float(x),
                        float(y),
                        float(width),
                        float(height),
                    ),
                )
            )

        self.previous_gray = gray

        detections.sort(
            key=lambda item: (
                item.bbox.x,
                item.bbox.y,
            )
        )

        return DetectionValidator.validate(image, detections)
