"""Threshold calibration on validation VIDEOS, never the held-out test split.

Internal risk goal is declared, not misrepresented as a hackathon requirement.
Observed false-alarm rate is per VIDEO, not correlated video frames.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json

from .inventory import hash_file,json_dump_atomic
from .perception import model_capability, normalize_label, TARGET, Box, iou
from .benchmark import _model_predictions, _names


def select_thresholds(rows: list[dict], *, max_negative_sequence_rate: float=.05,
                      min_positive_frames: int=30, min_negative_sequences: int=10) -> dict:
    if not 0 < max_negative_sequence_rate < 1:
        raise ValueError('Invalid internal FPR target')
    results={}
    for target in ('DRONE','BIRD'):
        positives=[]
        negative_by_seq=defaultdict(list)
        for row in rows:
            if row['split']!='val':
                raise ValueError('Calibration requires ONLY validation split')
            gt=normalize_label(row['ground_truth_class'])
            gt_box=Box(*row['ground_truth_bbox']) if row.get('ground_truth_bbox') is not None else None
            preds=[d for d in row['predictions'] if normalize_label(d['class'])==target]
            if gt==target and gt_box is not None:
                matched=[float(d['confidence']) for d in preds if iou(gt_box,Box(*d['bbox']))>=.5]
                positives.append(max(matched,default=0.))
            elif gt!=target:
                negative_by_seq[row['sequence_id']].append(max([float(d['confidence']) for d in preds],default=0.))
        negative_max=[max(v) for v in negative_by_seq.values()]
        candidates=sorted({.10,1.0,*positives,*negative_max})
        chosen=None; rec=0; rate=1.
        for threshold in candidates:
            neg_rate=sum(c>=threshold for c in negative_max)/len(negative_max) if negative_max else 1.
            positive_recall=sum(c>=threshold for c in positives)/len(positives) if positives else 0.
            if neg_rate <= max_negative_sequence_rate:
                chosen=threshold;rec=positive_recall;rate=neg_rate;break
        enough=len(positives)>=min_positive_frames and len(negative_max)>=min_negative_sequences
        results[target]={
            'threshold':chosen,
            'positive_frames':len(positives), 'negative_sequences':len(negative_max),
            'matched_positive_recall_at_threshold':rec if chosen is not None else None,
            'observed_negative_video_false_alarm_rate':rate if chosen is not None else None,
            'status':'VALIDATION_CALIBRATED' if chosen is not None and enough else 'BLOCKED_INSUFFICIENT_VALIDATION',
        }
    return {'method':'per-video-negative-max-confidence + per-frame-positive-IoU>=0.5',
            'split':'val','internal_max_negative_sequence_rate':max_negative_sequence_rate,
            'criteria':results,'status':'VALIDATION_CALIBRATED' if all(v['status']=='VALIDATION_CALIBRATED' for v in results.values())
            else 'BLOCKED_INSUFFICIENT_VALIDATION',
            'warning':'A calibrated validation threshold does NOT guarantee generalization. Test split is untouched.'}


def calibrate_exported_dataset(dataset_root,weights,output_file,*,max_frames=0,
                              max_negative_sequence_rate=.05) -> dict:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install opencv-python and ultralytics') from exc
    root=Path(dataset_root); weights=Path(weights)
    all_rows=[json.loads(s) for s in (root/'ground_truth.jsonl').read_text(encoding='utf-8').splitlines() if s]
    rows=sorted((r for r in all_rows if r['split']=='val'),
                key=lambda r:(r['sequence_id'],r['frame_index']))
    if max_frames:rows=rows[:max_frames]
    if not rows:raise ValueError('No validation frames')
    model=YOLO(str(weights))
    cap=model_capability(_names(model))
    if not cap['open_set_candidate']:
        raise ValueError('Calibration requires explicit negative classes; legacy 2-class model blocked')
    observations=[]
    for row in rows:
        frame=cv2.imread(str(root/row['image']))
        if frame is None:raise ValueError('Missing exported image: '+row['image'])
        candidates=_model_predictions(model,frame,low_conf=.10,imgsz=640)
        bbox=None
        if row['bbox_yolo'] is not None:
            x,y,w,h=row['bbox_yolo'];W,H=row['width'],row['height']
            bbox=[(x-w/2)*W,(y-h/2)*H,w*W,h*H]
        observations.append({'split':'val','sequence_id':row['sequence_id'],
             'ground_truth_class':row['class'] if bbox else 'BACKGROUND',
             'ground_truth_bbox':bbox,
             'predictions':[{'class':normalize_label(x.label),'bbox':x.box.to_list(),
                             'confidence':x.confidence} for x in candidates]})
    result=select_thresholds(observations,max_negative_sequence_rate=max_negative_sequence_rate)
    result['model_sha256']=hash_file(weights)
    result['validation_images']=len(rows)
    result['dataset_export_report_sha256']=(hash_file(root/'export_report.json') if
                                           (root/'export_report.json').exists() else None)
    json_dump_atomic(result,output_file)
    return result
