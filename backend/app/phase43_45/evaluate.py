"""Real detection/identity evaluation from independent GT/prediction frames.

No GT means NO accuracy claims. Require held-out source split.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import json
from .perception import Box, iou, normalize_label, TARGET
from .inventory import json_dump_atomic


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as src:
        return [json.loads(line) for line in src if line.strip()]


def _key(row: dict) -> tuple[str,int]:
    return str(row['sequence_id']), int(row['frame_index'])


def _det(d: dict) -> tuple[str,Box]:
    return normalize_label(d['class']), Box(*d['bbox'])


def evaluate_groundtruth(gt_rows: list[dict], pred_rows: list[dict],
                         *, iou_threshold: float = 0.5) -> dict:
    if not 0 < iou_threshold <= 1:
        raise ValueError('Invalid IoU')
    if not gt_rows:
        raise ValueError('Empty ground truth is not an evaluation')
    gt = {_key(row):row for row in gt_rows}
    pr = {_key(row):row for row in pred_rows}
    if len(gt) != len(gt_rows) or len(pr) != len(pred_rows):
        raise ValueError('Duplicate frames in groundtruth/predictions')
    if set(pr) != set(gt):
        raise ValueError('Frame mismatch between GT and predictions')
    splits = {row.get('split') for row in gt_rows}
    if splits != {'test'}:
        raise ValueError('Accuracy report requires ONLY held-out test split')
    tp, fp, fn = Counter(), Counter(), Counter()
    class_errors = Counter()
    negative_count = 0
    negative_target_fp = 0
    correct_tracks: dict[str,dict[str,Counter]] = defaultdict(lambda: defaultdict(Counter))
    gt_tot, pred_tot = Counter(), Counter()
    gt_seen_last = {}
    id_switches = 0
    matches_count = 0
    for key in sorted(gt):
        truth = gt[key].get('objects', [])
        predictions = pr[key].get('objects', [])
        g, p = [(_det(x),x) for x in truth], [(_det(x),x) for x in predictions]
        for (cl,_),entry in g:
            if 'track_id' in entry and entry['track_id'] is not None and cl in TARGET:
                gt_tot[(key[0],str(entry['track_id']))] += 1
        for (cl,_),entry in p:
            if 'track_id' in entry and entry['track_id'] is not None and cl in TARGET:
                pred_tot[(key[0],str(entry['track_id']))] += 1
        proposals = sorted(((iou(gg[1],pp[1]),i,j) for i,(gg,_) in enumerate(g)
                            for j,(pp,_) in enumerate(p)),reverse=True)
        used_gt, used_pred = set(), set()
        for overlap, i, j in proposals:
            if overlap < iou_threshold or i in used_gt or j in used_pred:
                continue
            used_gt.add(i); used_pred.add(j)
            (gcls,_), gt_det = g[i]
            (pcls,_), pd_det = p[j]
            matches_count += 1
            if gcls == pcls:
                tp[gcls] += 1
                gt_id, pred_id = gt_det.get('track_id'),pd_det.get('track_id')
                if (gcls in TARGET and gt_id is not None and pred_id is not None):
                    gt_id, pred_id = str(gt_id),str(pred_id)
                    correct_tracks[key[0]][gt_id][pred_id] += 1
                    prev = gt_seen_last.get((key[0],gt_id))
                    if prev is not None and prev != pred_id:
                        id_switches += 1
                    gt_seen_last[(key[0],gt_id)] = pred_id
            else:
                # A class confusion is both FN for true class and FP for predicted.
                fn[gcls] += 1
                fp[pcls] += 1
                class_errors[f'{gcls}->{pcls}'] += 1
        for i,((cls,_),_) in enumerate(g):
            if i not in used_gt:
                fn[cls] += 1
        for i,((cls,_),_) in enumerate(p):
            if i not in used_pred:
                fp[cls] += 1
        if not any(c in TARGET for (c,_),_ in g):
            negative_count += 1
            if any(c == 'DRONE' for (c,_),_ in p):
                negative_target_fp += 1
    classes = sorted(set(tp)|set(fp)|set(fn))
    per_class = {}
    for cls in classes:
        den_p, den_r = tp[cls]+fp[cls],tp[cls]+fn[cls]
        per_class[cls] = {'tp':tp[cls],'fp':fp[cls],'fn':fn[cls],
                          'precision':tp[cls]/den_p if den_p else None,
                          'recall':tp[cls]/den_r if den_r else None}
    idf1, idtp = None, None
    if gt_tot or pred_tot:
        try:
            from scipy.optimize import linear_sum_assignment
            gt_ids = sorted(gt_tot)
            pd_ids = sorted(pred_tot)
            import numpy as np
            contingency = np.zeros((len(gt_ids),len(pd_ids)),dtype=int)
            gidx, pidx = {v:i for i,v in enumerate(gt_ids)}, {v:i for i,v in enumerate(pd_ids)}
            for seq, groups in correct_tracks.items():
                for gid, pmap in groups.items():
                    for pid, count in pmap.items():
                        contingency[gidx[(seq,gid)],pidx[(seq,pid)]] += count
            if contingency.size:
                rows,cols = linear_sum_assignment(-contingency)
                idtp = int(sum(contingency[i,j] for i,j in zip(rows,cols)))
            else:
                idtp = 0
            total = sum(gt_tot.values())+sum(pred_tot.values())
            idf1 = (2*idtp/total) if total else None
        except ImportError:
            pass
    return {'status':'MEASURED_OFFLINE_NOT_FIELD_VALIDATED',
            'frames':len(gt), 'matched_objects':matches_count,
            'per_class':per_class, 'class_confusion':dict(class_errors),
            'negative_frames':negative_count,
            'negative_frames_predicted_drone':negative_target_fp,
            'negative_drone_false_alarm_frame_rate':negative_target_fp/negative_count if negative_count else None,
            'id_switches_observed':id_switches,
            'idf1':idf1, 'idtp':idtp,
            'HOTA':None, 'HOTA_status':'BLOCKED: use MOT GT + TrackEval HOTA implementation',
            'mAP50':None, 'mAP50_95':None,
            'mAP_status':'BLOCKED: use YOLO validation or official COCOeval protocol',
            'limits':['IoU-matched single test split',
                      'IDF1 requires reference and prediction track_id for both sides',
                      'Halmstad is single-object per sequence; multiobject MOT requires separate data']}
