"""Compatibility adapters for existing SkyGuard engine APIs.

Explicit integration only; imports existing classes lazily. DOES NOT alter
app/main.py or existing camera session wiring on installation.
"""
from __future__ import annotations
from pathlib import Path
import math

from .perception import DecisionGate, Box, Candidate, normalize_label


class FourClassSkyGuardDetector:
    """Compatible detector.detect(np.ndarray) -> app.detection.engine.Detection[].

    A four-class (drone/bird/airplane/helicopter) or three-class
    (drone/bird/other) model is mandatory. Legacy 2-class model is refused.
    AIRPLANE/HELICOPTER/OTHER become untracked UNKNOWN objects.
    """
    skyguard_requires_target_only_tracking = True

    def __init__(self, weights_path, *, low_confidence: float=.10,
                 target_threshold: float=.60, imgsz: int=640, device='cpu'):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('ultralytics is required for production adapter') from exc
        if not 0 < low_confidence <= 1 or not 0 < target_threshold <= 1:
            raise ValueError('Invalid confidence')
        if imgsz < 32:
            raise ValueError('Invalid image size')
        self.model=YOLO(str(Path(weights_path).resolve(strict=True)))
        names=self.model.names
        self.names={int(k):str(v) for k,v in names.items()} if isinstance(names,dict) else dict(enumerate(names))
        self.gate=DecisionGate(self.names,{'DRONE':target_threshold,'BIRD':target_threshold})
        if not self.gate.capability['open_set_candidate']:
            raise ValueError('Cannot use two-class YOLO in open-set production adapter')
        self.low_confidence=low_confidence
        self.imgsz=imgsz
        self.device=device

    def reset(self):
        pass

    def detect(self, frame_image):
        from app.detection.engine import BoundingBox, Detection
        height,width=frame_image.shape[:2]
        raw=self.model.predict(source=frame_image,conf=self.low_confidence,imgsz=self.imgsz,
                               device=self.device,verbose=False)
        if not raw:
            return []
        result=[]
        for candidate in raw[0].boxes:
            class_id=int(candidate.cls[0].item())
            if class_id not in self.names:
                raise ValueError(f'Unknown YOLO class index {class_id}')
            x1,y1,x2,y2=(float(v) for v in candidate.xyxy[0].tolist())
            x1=max(0.,x1);y1=max(0.,y1)
            x2=min(float(width),x2);y2=min(float(height),y2)
            if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or x2<=x1 or y2<=y1:
                raise ValueError('Invalid YOLO bbox')
            score=float(candidate.conf[0].item())
            accepted=self.gate.decide(Candidate(self.names[class_id],score,
                                                Box(x1,y1,x2-x1,y2-y1)))
            result.append(Detection(bbox=BoundingBox(x=x1,y=y1,width=x2-x1,height=y2-y1),
                                    object_class=accepted['class'].lower(),
                                    confidence=score))
        return result


class TargetOnlyTrackerProxy:
    """Conservative wrapper around any *existing* SkyGuard tracker.

    Keeps the existing tracker algorithm and result types, but never forwards
    UNKNOWN, AIRPLANE, HELICOPTER, or malformed classes to update().
    """
    def __init__(self, tracker):
        self.tracker=tracker
        self.last_unknown_count=0

    def reset(self):
        self.last_unknown_count=0
        return self.tracker.reset()

    def update(self, frame_index, detections):
        target=[]
        for item in detections:
            raw=getattr(item,'object_class',None)
            if isinstance(raw,str) and normalize_label(raw) in ('DRONE','BIRD'):
                target.append(item)
        self.last_unknown_count=len(detections)-len(target)
        return self.tracker.update(frame_index=frame_index,detections=target)
