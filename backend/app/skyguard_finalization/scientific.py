"""Reproducible ground-truth scoring. No inferred ground truth or GPS."""
from __future__ import annotations
import math
from collections import defaultdict

TARGETS = {"DRONE", "BIRD"}
CLASSES = {"DRONE", "BIRD", "UNKNOWN"}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if not all(math.isfinite(float(v)) for v in (*a, *b)):
        raise ValueError("Nonfinite bbox")
    if min(a[2]-a[0],a[3]-a[1],b[2]-b[0],b[3]-b[1]) <= 0:
        raise ValueError("Invalid bbox")
    inter = max(0., x2-x1)*max(0., y2-y1)
    union = ((a[2]-a[0])*(a[3]-a[1]) +
             (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.


def label(obj) -> str:
    value = str(obj.get('class',obj.get('classification',''))).upper()
    if value not in CLASSES:
        raise ValueError(f"Invalid annotated class: {value}")
    return value


def validate_samples(rows, *, predictions=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Non-empty list of labelled frame records required')
    seen = set()
    for r in rows:
        seq = r.get('sequence_id')
        frame = r.get('frame_index')
        if not isinstance(seq, str) or not seq or type(frame) is not int or frame < 0:
            raise ValueError('sequence_id and nonnegative integer frame_index required')
        key = (seq, frame)
        if key in seen: raise ValueError('Duplicate sequence/frame')
        seen.add(key)
        obs = r.get('objects')
        if not isinstance(obs, list): raise ValueError('objects list required')
        ids = set()
        for o in obs:
            cls = label(o)
            bbox = o.get('bbox')
            if not isinstance(bbox,(list,tuple)) or len(bbox)!=4:
                raise ValueError('XYXY bbox required')
            iou(bbox,bbox)
            tid=o.get('track_id')
            if cls == 'UNKNOWN' and tid is not None:
                raise ValueError('UNKNOWN cannot carry tracking identity')
            if tid is not None:
                identity=(cls,str(tid))
                if identity in ids: raise ValueError('Duplicate ID within frame')
                ids.add(identity)
    return { (r['sequence_id'],r['frame_index']): r['objects'] for r in rows }


def score_ground_truth(gt, preds, threshold=0.5):
    """Class-aware IoU assignment; global IDF1 only if GT and predictions both carry track IDs.

    Requires source-disjoint, human-reviewed GT for a meaningful claim. This
    implementation intentionally does NOT claim HOTA or geometric accuracy.
    """
    if not (0 < threshold <= 1): raise ValueError('invalid threshold')
    from scipy.optimize import linear_sum_assignment
    reference = validate_samples(gt)
    prediction = validate_samples(preds,predictions=True)
    if set(prediction) - set(reference):
        raise ValueError('Prediction frames outside labeled ground truth')
    tp = fp = fn = 0
    matrix = defaultdict(int)
    class_confusion = defaultdict(int)
    gt_id_counts = defaultdict(int)
    pred_id_counts = defaultdict(int)
    pairs = defaultdict(int)
    gt_all_id = pred_all_id = True
    for key, truth in sorted(reference.items()):
        estim = prediction.get(key, [])
        for o in truth:
            if label(o) in TARGETS:
                if o.get('track_id') is None: gt_all_id = False
                else: gt_id_counts[(key[0],label(o),str(o['track_id']))] += 1
        for o in estim:
            if label(o) in TARGETS:
                if o.get('track_id') is None: pred_all_id = False
                else: pred_id_counts[(key[0],label(o),str(o['track_id']))] += 1
        # Detection scoring includes UNKNOWN, but association is class-aware.
        n,m=len(truth),len(estim)
        weights=[[iou(a['bbox'],b['bbox']) if label(a)==label(b) else 0.
                  for b in estim] for a in truth]
        matched_g=set(); matched_p=set()
        if n and m:
            rows,cols=linear_sum_assignment([[-w for w in row] for row in weights])
            for a,b in zip(rows,cols):
                if weights[a][b] < threshold: continue
                matched_g.add(a); matched_p.add(b);tp+=1
                matrix[(label(truth[a]),label(estim[b]))] += 1
                a_id,b_id=truth[a].get('track_id'),estim[b].get('track_id')
                if a_id is not None and b_id is not None and label(truth[a]) in TARGETS:
                    pairs[(key[0],label(truth[a]),str(a_id),str(b_id))] += 1
        # Separate CLASSIFICATION confusion from class-aware detection TP.
        # Spatial matching can expose BIRD->DRONE false alarms rather than
        # hiding them as unmatched class-aware pairs.
        spatial_matched_g=set(); spatial_matched_p=set()
        if n and m:
            spatial=[[iou(a['bbox'],b['bbox']) for b in estim] for a in truth]
            ri,ci=linear_sum_assignment([[-v for v in row] for row in spatial])
            for a,b in zip(ri,ci):
                if spatial[a][b] >= threshold:
                    spatial_matched_g.add(a); spatial_matched_p.add(b)
                    class_confusion[f"{label(truth[a])}->{label(estim[b])}"]+=1
        for idx,o in enumerate(truth):
            if idx not in spatial_matched_g:
                class_confusion[f"{label(o)}->MISSED"]+=1
        for idx,o in enumerate(estim):
            if idx not in spatial_matched_p:
                class_confusion[f"BACKGROUND->{label(o)}"]+=1
        fn += n-len(matched_g); fp += m-len(matched_p)
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    idf1=None
    if gt_all_id and pred_all_id and (gt_id_counts or pred_id_counts):
        # Assignment must be global across entire sequence, not frame-wise.
        keys_g=list(gt_id_counts); keys_p=list(pred_id_counts)
        weights=[[pairs.get((g[0],g[1],g[2],p[2]),0)
                  if g[:2]==p[:2] else 0 for p in keys_p] for g in keys_g]
        if keys_g and keys_p:
            ii,jj=linear_sum_assignment([[-w for w in row] for row in weights])
            idtp=sum(weights[i][j] for i,j in zip(ii,jj))
        else: idtp=0
        total_gt=sum(gt_id_counts.values()); total_pred=sum(pred_id_counts.values())
        idfn=total_gt-idtp; idfp=total_pred-idtp
        idf1=(2*idtp)/(2*idtp+idfn+idfp) if (2*idtp+idfn+idfp)>0 else None
    return {
        'ground_truth_frames':len(reference), 'tp':tp,'fp':fp,'fn':fn,
        'precision':precision,'recall':recall,'idf1':idf1,
        'idf1_available':idf1 is not None,
        'matched_by_class':{f'{a}:{b}':v for (a,b),v in sorted(matrix.items())},
        'class_confusion':dict(sorted(class_confusion.items())),
        'hota':None, 'hota_status':'NOT_IMPLEMENTED_NO_CLAIM',
        'ground_truth_warning':'Source-disjoint split and annotation/license rights must be independently verified',
    }
