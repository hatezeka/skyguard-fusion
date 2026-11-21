"""Combine specialized detections with motion-based unknown candidates."""

from app.detection.engine import DetectionEngine, DetectionValidator
from app.detection.tracker import box_iou


class AirborneFusionDetectionEngine(DetectionEngine):
    def __init__(
        self,
        specialized_detector,
        candidate_detector,
        overlap_threshold=0.3,
    ):
        if not 0 < overlap_threshold <= 1:
            raise ValueError("Invalid overlap threshold")

        self.specialized_detector = specialized_detector
        self.candidate_detector = candidate_detector
        self.overlap_threshold = overlap_threshold

    def reset(self):
        for detector in (
            self.specialized_detector,
            self.candidate_detector,
        ):
            reset = getattr(detector, "reset", None)
            if callable(reset):
                reset()

    def detect(self, image):
        DetectionValidator.validate(image, [])

        specialized = self.specialized_detector.detect(image)
        candidates = self.candidate_detector.detect(image)

        DetectionValidator.validate(image, specialized)
        DetectionValidator.validate(image, candidates)

        results = list(specialized)

        for candidate in candidates:
            if candidate.object_class != "unknown":
                continue

            overlaps = any(
                box_iou(candidate.bbox, existing.bbox)
                >= self.overlap_threshold
                for existing in results
            )

            if not overlaps:
                results.append(candidate)

        return DetectionValidator.validate(image, results)
