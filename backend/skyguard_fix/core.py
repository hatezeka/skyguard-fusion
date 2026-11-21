"""Scientifically bounded inference: multi-resolution detections, no fabricated UNKNOWN."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Protocol
import math
import numpy as np

LABELS = {'DRONE', 'BIRD', 'UNKNOWN'}

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    pass_name: str
    track_id: int | None = None
    verdict_status: str = "SINGLE_MODEL_PREDICTION"
    vote_evidence: tuple[tuple[str, float], ...] = ()
    merged_proposals: int = 1

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError('Unsupported class')
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError('Invalid confidence')
        if not all(math.isfinite(v) for v in self.xyxy):
            raise ValueError('Invalid box')
        x1,y1,x2,y2=self.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Degenerate box')
        if self.label == 'UNKNOWN' and self.track_id is not None:
            raise ValueError('UNKNOWN must not have a track ID')

class Predictor(Protocol):
    def predict(self, image: np.ndarray, imgsz: int, conf: float): ...

def overlap(a: Detection, b: Detection) -> float:
    x1=max(a.xyxy[0],b.xyxy[0]);y1=max(a.xyxy[1],b.xyxy[1])
    x2=min(a.xyxy[2],b.xyxy[2]);y2=min(a.xyxy[3],b.xyxy[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=(a.xyxy[2]-a.xyxy[0])*(a.xyxy[3]-a.xyxy[1])
    area_b=(b.xyxy[2]-b.xyxy[0])*(b.xyxy[3]-b.xyxy[1])
    return inter/(area_a+area_b-inter) if area_a+area_b-inter>0 else 0

def _containment(a: Detection, b: Detection) -> float:
    x1=max(a.xyxy[0],b.xyxy[0]);y1=max(a.xyxy[1],b.xyxy[1])
    x2=min(a.xyxy[2],b.xyxy[2]);y2=min(a.xyxy[3],b.xyxy[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    areas=[(d.xyxy[2]-d.xyxy[0])*(d.xyxy[3]-d.xyxy[1]) for d in (a,b)]
    return inter/max(1e-9,min(areas))


def _same_object(a: Detection, b: Detection, iou_threshold: float=.22,
                 containment_threshold: float=.72) -> bool:
    # Cross-class geometric association. Different labels do not prevent merging.
    # The containment gate is useful for full-body/partial-wing proposals.
    return overlap(a,b)>=iou_threshold or _containment(a,b)>=containment_threshold


def merge_object_proposals(detections: list[Detection], iou_threshold:float=.22,
                           containment_threshold:float=.72,
                           vote_margin:float=.18) -> list[Detection]:
    """One geometric object -> one box + one auditable tentative verdict.

    Votes are *not* independent probability estimates. Each pass/class only
    contributes its strongest proposal to prevent duplicate boxes from
    artificially multiplying confidence. Ambiguous outcomes remain marked
    AMBIGUOUS, never fabricated as a trained UNKNOWN class.
    """
    if not 0 < iou_threshold < 1 or not 0 < containment_threshold <= 1:
        raise ValueError('Invalid geometric thresholds')
    if not 0 <= vote_margin < 1:raise ValueError('Invalid vote margin')
    if not detections:return []
    # Strong-first greedy clustering limits transitive bridges between neighbors.
    ordered=sorted(detections,key=lambda d:(-d.confidence,d.xyxy))
    clusters: list[list[Detection]]=[]
    for d in ordered:
        eligible=[]
        for index,group in enumerate(clusters):
            # Must overlap a high-confidence geometric anchor, not any member;
            # prevents A~B~C chaining from merging two separate birds.
            anchor=group[0]
            if _same_object(d,anchor,iou_threshold,containment_threshold):
                eligible.append((overlap(d,anchor),_containment(d,anchor),index))
        if eligible:
            idx=max(eligible)[2]
            clusters[idx].append(d)
        else:
            clusters.append([d])
    merged=[]
    for group in clusters:
        votes:dict[str,dict[str,float]]={}
        for d in group:
            votes.setdefault(d.label,{})[d.pass_name]=max(d.confidence,votes.get(d.label,{}).get(d.pass_name,0.))
        # Balance class votes across passes, cap to one contribution per pass.
        score={label:sum(by_pass.values())/max(1,len(by_pass)) for label,by_pass in votes.items()}
        ranking=sorted(score.items(),key=lambda kv:(-kv[1],kv[0]))
        best_label,best_score=ranking[0]
        runner=ranking[1][1] if len(ranking)>1 else 0.
        status='CONFLICT_REVIEW' if runner>0 and best_score-runner<vote_margin else 'CONSENSUS'
        # Representative box is the proposal most geometrically supported by
        # its peers, not a massive union of boxes across the sky.
        def geometry_support(d):
            return (sum(overlap(d,g) for g in group if g is not d),d.confidence)
        representative=max(group,key=geometry_support)
        merged.append(Detection(
            label=best_label,confidence=best_score,xyxy=representative.xyxy,
            pass_name='consensus',track_id=None,verdict_status=status,
            vote_evidence=tuple(ranking),merged_proposals=len(group)))
    return sorted(merged,key=lambda d:(-d.confidence,d.xyxy))


def class_aware_nms(detections: list[Detection], iou:float=.5)->list[Detection]:
    """Legacy interface retained for other modules; class-specific behavior."""
    if not 0<iou<1:raise ValueError('iou must be in (0,1)')
    out=[]
    for d in sorted(detections,key=lambda o:-o.confidence):
        if not any(k.label==d.label and overlap(d,k)>iou for k in out):out.append(d)
    return out

def tiles(width:int,height:int,tile:int=640,overlap_fraction:float=.20):
    if width<=0 or height<=0 or tile<32 or not 0<=overlap_fraction<1:raise ValueError('Bad tile parameters')
    def starts(length):
        if length<=tile:return [0]
        last=length-tile
        step=max(1,int(tile*(1-overlap_fraction)))
        return sorted(set(list(range(0,last+1,step))+[last]))
    for y in starts(height):
        for x in starts(width):
            yield (x,y,min(width,x+tile),min(height,y+tile))

def _decode_raw(preds, labels, offset=(0,0), pass_name='large'):
    ox,oy=offset
    for cls,conf,box in preds:
        label=labels[int(cls)].upper()
        if label not in LABELS:raise ValueError(f'Unknown model class: {label}')
        x1,y1,x2,y2=[float(n) for n in box]
        yield Detection(label,float(conf),(x1+ox,y1+oy,x2+ox,y2+oy),pass_name)

def multiscale(frame:np.ndarray, predictor:Predictor, labels:dict[int,str], confidence:float=.25,
               high_resolution:int=1280, tile_size:int=640, include_tiles:bool=False,
               mode:str="adaptive"):
    if frame is None or frame.ndim != 3 or frame.shape[2]!=3:raise ValueError('Expected BGR HxWx3')
    if not 0 < confidence < 1: raise ValueError('confidence must be in (0,1)')
    h,w=frame.shape[:2]
    all_dets=[]
    if mode not in ('adaptive','full'):raise ValueError('Invalid mode')
    first=list(_decode_raw(predictor.predict(frame,640,confidence),labels,pass_name='large'))
    all_dets.extend(first)
    # Adaptive fast path: one 640 inference for a clear object; escalate for
    # missing/low-confidence/small objects, or explicitly request exhaustive mode.
    uncertain=(not first or max(d.confidence for d in first)<.55 or
               any((d.xyxy[2]-d.xyxy[0])*(d.xyxy[3]-d.xyxy[1])<.001*w*h for d in first))
    if mode=='full' or uncertain:
        all_dets.extend(_decode_raw(predictor.predict(frame,high_resolution,confidence),labels,pass_name='medium'))
    if include_tiles and (mode=='full' or uncertain):
        for x1,y1,x2,y2 in tiles(w,h,tile_size):
            patch=frame[y1:y2,x1:x2]
            all_dets.extend(_decode_raw(predictor.predict(patch,640,confidence),labels,(x1,y1),'small'))
    bounded=[]
    for d in all_dets:
        x1,y1,x2,y2=d.xyxy
        x1=max(0,min(w,float(x1)));x2=max(0,min(w,float(x2)))
        y1=max(0,min(h,float(y1)));y2=max(0,min(h,float(y2)))
        if x2>x1 and y2>y1:
            bounded.append(Detection(d.label,d.confidence,(x1,y1,x2,y2),d.pass_name))
    return merge_object_proposals(bounded)

class UltralyticsPredictor:
    def __init__(self,weights):
        from ultralytics import YOLO
        self.model=YOLO(str(weights))
        self.names={int(k):str(v).upper() for k,v in self.model.names.items()}
        classes=set(self.names.values())
        if classes not in ({'DRONE','BIRD'},{'DRONE','BIRD','UNKNOWN'}):
            raise ValueError(f'Unexpected checkpoint labels: {self.names}')
        self.is_v2=('UNKNOWN' in classes)
    def predict(self,image,imgsz,conf):
        results=self.model.predict(image,imgsz=imgsz,conf=conf,verbose=False)
        for box in results[0].boxes:
            yield (int(box.cls.item()),float(box.conf.item()),tuple(float(v) for v in box.xyxy[0].tolist()))

def safe_label(det:Detection,labels:dict[int,str]):
    """No relabeling of V1 predicted DRONE/BIRD as UNKNOWN just from confidence."""
    return det.label
