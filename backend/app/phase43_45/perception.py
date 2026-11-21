"""Open-set fail-closed decision boundary and a reference target-only tracker.

The tracker below is for offline evidence/replay; do not silently replace the
production tracker without explicit integration/regression testing.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

TARGET = frozenset({'DRONE', 'BIRD'})
REJECT = frozenset({'AIRPLANE', 'HELICOPTER', 'OTHER', 'UNKNOWN'})


def normalize_label(label: str) -> str:
    aliases = {'UAV': 'DRONE', 'QUADCOPTER': 'DRONE', 'BIRDS': 'BIRD',
               'PLANE': 'AIRPLANE', 'AIRCRAFT': 'AIRPLANE', 'HELI': 'HELICOPTER', 'OTHER_AIRBORNE': 'OTHER',
               'OTHER-AIRBORNE': 'OTHER'}
    result = str(label).strip().upper()
    return aliases.get(result, result)


def model_capability(names: list[str] | dict[int, str]) -> dict:
    given = {normalize_label(x) for x in (names.values() if isinstance(names, dict) else names)}
    ready = TARGET.issubset(given) and (('OTHER' in given) or
            {'AIRPLANE', 'HELICOPTER'}.issubset(given))
    return {'classes': sorted(given), 'open_set_candidate': ready,
            'reason': 'Has explicit negative classes' if ready else
            'No explicit airborne negative class: 2-class predictions cannot validate UNKNOWN'}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    def validate(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.w, self.h)):
            raise ValueError('Nonfinite bbox')
        if self.w <= 0 or self.h <= 0:
            raise ValueError('Non-positive bbox')

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]


@dataclass(frozen=True)
class Candidate:
    label: str
    confidence: float
    box: Box

    def validate(self) -> None:
        self.box.validate()
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError('Invalid confidence')


def iou(a: Box, b: Box) -> float:
    xx1, yy1 = max(a.x, b.x), max(a.y, b.y)
    xx2, yy2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0., xx2-xx1) * max(0., yy2-yy1)
    union = a.w*a.h+b.w*b.h-inter
    return inter / union if union > 0 else 0.


class DecisionGate:
    def __init__(self, names: list[str] | dict[int, str],
                 thresholds: dict[str, float] | None = None,
                 *, require_negative_class: bool = True):
        self.capability = model_capability(names)
        self.require_negative_class = require_negative_class
        thresholds = thresholds or {'DRONE': 0.60, 'BIRD': 0.60}
        self.thresholds = {}
        for cls in TARGET:
            value = float(thresholds.get(cls, 0.60))
            if not 0 < value <= 1:
                raise ValueError('Invalid threshold')
            self.thresholds[cls] = value

    def decide(self, candidate: Candidate) -> dict:
        candidate.validate()
        original = normalize_label(candidate.label)
        reject_reason = None
        if original not in TARGET:
            reject_reason = 'NON_TARGET_OR_UNKNOWN'
        elif self.require_negative_class and not self.capability['open_set_candidate']:
            reject_reason = 'TWO_CLASS_MODEL_UNSAFE_FOR_OPEN_SET'
        elif candidate.confidence < self.thresholds[original]:
            reject_reason = 'BELOW_PROVISIONAL_THRESHOLD'
        label = original if reject_reason is None else 'UNKNOWN'
        return {'class': label, 'raw_class': original,
                'confidence': candidate.confidence, 'bbox': candidate.box.to_list(),
                'target_eligible': label in TARGET,
                'reject_reason': reject_reason,
                'thresholds_calibrated': False}


@dataclass
class _Track:
    track_id: int
    cls: str
    box: Box
    last_index: int
    hits: int


class TargetOnlyTracker:
    """Reference continuity tracker with conservative ambiguity rejection.

    Tentative tracks have internal IDs but IDs are NOT emitted to API/evidence
    until confirmed. UNKNOWN NEVER enters the tracker. Ambiguity suppresses
    current-frame IDs; a production-grade tracker can use stronger association.
    """
    def __init__(self, min_hits: int = 2, iou_threshold: float = .30,
                 max_gap: int = 3, ambiguity_delta: float = .08):
        if min_hits < 1 or max_gap < 0 or not 0 < iou_threshold <= 1 or ambiguity_delta < 0:
            raise ValueError('Invalid tracker parameters')
        self.min_hits, self.iou_threshold = min_hits, iou_threshold
        self.max_gap, self.ambiguity_delta = max_gap, ambiguity_delta
        self.tracks: dict[int, _Track] = {}
        self.next_id = 1
        self.last_index = -1

    def reset(self) -> None:
        self.tracks = {}
        self.next_id = 1
        self.last_index = -1

    def update(self, frame_index: int, detections: list[dict]) -> dict:
        if frame_index <= self.last_index:
            raise ValueError('Frame indices must be strictly monotonic')
        self.last_index = frame_index
        self.tracks = {k:v for k,v in self.tracks.items()
                       if frame_index - v.last_index - 1 <= self.max_gap}
        clean = []
        for d in detections:
            cls = normalize_label(d.get('class', 'UNKNOWN'))
            if cls not in TARGET or not d.get('target_eligible', False):
                continue
            box = Box(*d['bbox'])
            box.validate()
            clean.append((cls, box, float(d['confidence'])))

        candidates: list[tuple[float,int,int]] = []
        by_det: dict[int,list[tuple[float,int]]] = {}
        by_track: dict[int,list[tuple[float,int]]] = {}
        for tid, track in self.tracks.items():
            for i, (cls, box, _) in enumerate(clean):
                if cls != track.cls:
                    continue
                score = iou(track.box, box)
                if score >= self.iou_threshold:
                    candidates.append((score, tid, i))
                    by_det.setdefault(i, []).append((score, tid))
                    by_track.setdefault(tid, []).append((score, i))
        ambiguous_det, ambiguous_track = set(), set()
        for d_idx, options in by_det.items():
            s = sorted(options, reverse=True)
            if len(s) >= 2 and s[0][0]-s[1][0] <= self.ambiguity_delta:
                ambiguous_det.add(d_idx)
        for tid, options in by_track.items():
            s = sorted(options, reverse=True)
            if len(s) >= 2 and s[0][0]-s[1][0] <= self.ambiguity_delta:
                ambiguous_track.add(tid)
                ambiguous_det.update(i for _,i in options)

        matched_ids, matched_det = set(), set()
        confirmed = []
        for score, tid, idx in sorted(candidates, key=lambda x:(-x[0],x[1],x[2])):
            if (tid in matched_ids or idx in matched_det or
                    tid in ambiguous_track or idx in ambiguous_det):
                continue
            cls, box, conf = clean[idx]
            state = self.tracks[tid]
            state.box, state.last_index = box, frame_index
            state.hits += 1
            matched_ids.add(tid)
            matched_det.add(idx)
            if state.hits >= self.min_hits:
                confirmed.append({'track_id': tid, 'class': cls, 'bbox': box.to_list(),
                                  'confidence': conf, 'frame_index': frame_index})

        for idx, (cls, box, conf) in enumerate(clean):
            if idx in matched_det or idx in ambiguous_det:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = _Track(tid, cls, box, frame_index, 1)
            if self.min_hits <= 1:
                confirmed.append({'track_id': tid, 'class': cls, 'bbox': box.to_list(),
                                  'confidence': conf, 'frame_index': frame_index})
        return {'confirmed': sorted(confirmed, key=lambda x:x['track_id']),
                'ambiguous_count': len(ambiguous_det),
                'rejected_count': len(detections)-len(clean),
                'tentative_count': sum(t.hits < self.min_hits for t in self.tracks.values())}
