"""Actual YOLO inference on held-out exported frames, measured FPS and errors.

Does not claim HOTA, physical range, GPS, or USB field validation.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import perf_counter
from datetime import datetime, timezone
import json
import platform

from .inventory import hash_file, json_dump_atomic
from .evaluate import evaluate_groundtruth
from .perception import Box, Candidate, DecisionGate, TargetOnlyTracker, model_capability, normalize_label
from .maps import dual_scene


def percentile(samples: list[float], percentile_value: float) -> float | None:
    if not samples:
        return None
    x=sorted(samples)
    index=(len(x)-1)*percentile_value/100
    lo=int(index); hi=min(len(x)-1,lo+1)
    return x[lo]*(1-(index-lo))+x[hi]*(index-lo)


def _names(model) -> dict:
    names = model.names
    if isinstance(names, dict):
        return {int(k):str(v) for k,v in names.items()}
    return {i:str(v) for i,v in enumerate(names)}


def _model_predictions(model, image, *, low_conf: float, imgsz: int) -> list[Candidate]:
    result = model.predict(source=image, conf=low_conf, imgsz=imgsz, verbose=False)[0]
    names = _names(model)
    results=[]
    if result.boxes is None:
        return results
    for box in result.boxes:
        xyxy=box.xyxy[0].tolist()
        cls=int(box.cls[0].item())
        conf=float(box.conf[0].item())
        if cls not in names:
            raise ValueError(f'Out-of-range YOLO class ID: {cls}')
        results.append(Candidate(names[cls], conf, Box(float(xyxy[0]),float(xyxy[1]),
                                                     float(xyxy[2])-float(xyxy[0]),
                                                     float(xyxy[3])-float(xyxy[1]))))
    return results


def benchmark_exported_dataset(dataset_root: str | Path, weights: str | Path,
                               output_dir: str | Path, *, max_frames: int = 0,
                               imgsz: int = 640, low_conf: float = 0.10,
                               thresholds: dict | None = None,
                               calibration: dict | None = None) -> dict:
    if max_frames < 0 or imgsz < 32 or not 0 < low_conf <= 1:
        raise ValueError('Invalid benchmark parameters')
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install: pip install opencv-python ultralytics') from exc
    root=Path(dataset_root); out=Path(output_dir)
    if not (root/'ground_truth.jsonl').exists():
        raise FileNotFoundError('Export ground_truth.jsonl missing')
    weight=Path(weights)
    if not weight.is_file():
        raise FileNotFoundError(weight)
    gt_rows = [json.loads(line) for line in (root/'ground_truth.jsonl').read_text(encoding='utf-8').splitlines()
               if line.strip()]
    gt_rows = sorted((row for row in gt_rows if row['split']=='test'),
                     key=lambda x:(x['sequence_id'],x['frame_index']))
    if max_frames:
        gt_rows=gt_rows[:max_frames]
    if not gt_rows:
        raise ValueError('No test frames; export additional test sequences')
    model=YOLO(str(weight))
    names=_names(model)
    if calibration is not None:
        if calibration.get('model_sha256') != hash_file(weight):
            raise ValueError('Calibration is not tied to this checkpoint SHA256')
        if calibration.get('status') != 'VALIDATION_CALIBRATED':
            raise ValueError('Calibration is blocked/incomplete; cannot claim calibrated')
        thresholds={cls:float(calibration['criteria'][cls]['threshold']) for cls in ('DRONE','BIRD')}
    gate=DecisionGate(names,thresholds)
    tracker=TargetOnlyTracker(min_hits=2)
    old_seq=None
    gt_frames=[]; predictions=[]; scenes=[]; latencies=[]; io_time=[]
    classification_counts=Counter()
    t0=perf_counter()
    for row in gt_rows:
        if row['sequence_id']!=old_seq:
            tracker.reset();old_seq=row['sequence_id']
        name=root/row['image']
        start=perf_counter()
        image=cv2.imread(str(name))
        io_time.append(perf_counter()-start)
        if image is None:
            raise ValueError(f'Image failed to load: {name}')
        inf_start=perf_counter()
        raw=_model_predictions(model,image,low_conf=low_conf,imgsz=imgsz)
        decisions=[gate.decide(p) for p in raw]
        result=tracker.update(int(row['frame_index']),decisions)
        latencies.append(perf_counter()-inf_start)
        objects=[]
        for d in decisions:
            classification_counts[d['class']] += 1
            objects.append({'class':d['class'],'bbox':d['bbox'], 'confidence':d['confidence'],
                            'raw_class':d['raw_class'],'reject_reason':d['reject_reason']})
        # associate returned confirmed tracks with nearest matching decision bbox
        for t in result['confirmed']:
            for d in objects:
                if d['class']==t['class'] and d['bbox']==t['bbox'] and 'track_id' not in d:
                    d['track_id']=t['track_id'];break
        pixel_bin='NO_GT_BOX'
        if row['bbox_yolo'] is not None:
            _,_,bw,bh=row['bbox_yolo']
            pixel_area=bw*row['width']*bh*row['height']
            pixel_bin=('SMALL_LE_32SQ_PX' if pixel_area <=32**2 else
                      ('MEDIUM_LE_96SQ_PX' if pixel_area <=96**2 else 'LARGE_GT_96SQ_PX'))
        key={'sequence_id':row['sequence_id'], 'frame_index':int(row['frame_index']),
             'split':'test','modality':row['modality'],'apparent_size_bin':pixel_bin}
        gtobjs=[]
        if row['bbox_yolo'] is not None:
            x,y,w,h=row['bbox_yolo']; W,H=int(row['width']),int(row['height'])
            gtobjs=[{'class':row['class'], 'bbox':[(x-w/2)*W,(y-h/2)*H,w*W,h*H],
                     'track_id':str(row['gt_track_id'])}]
        gt_frames.append({**key,'objects':gtobjs})
        predictions.append({**key,'objects':objects})
        scene=dual_scene(row['sequence_id'],int(row['frame_index']),
                         float(row['frame_index'])*1000.0, result['confirmed'],image.shape[1])
        scenes.append(scene)
    duration=perf_counter()-t0
    measured=evaluate_groundtruth(gt_frames,predictions)
    # Stratify by independently supplied dataset modality and apparent box size,
    # never by guessed physical range/distance.
    slices={}
    for dimension in ('modality','apparent_size_bin'):
        for value in sorted({g[dimension] for g in gt_frames}):
            gs=[g for g in gt_frames if g[dimension]==value]
            selected={(g['sequence_id'],g['frame_index']) for g in gs}
            ps=[p for p in predictions if (p['sequence_id'],p['frame_index']) in selected]
            r=evaluate_groundtruth(gs,ps)
            slices[f'{dimension}:{value}']={
                 'frames':r['frames'],'per_class':r['per_class'],
                 'negative_drone_false_alarm_frame_rate':r['negative_drone_false_alarm_frame_rate']}
    measured['stratified_evidence']=slices
    measured['pixel_bin_definitions']='COCO-style 32x32 / 96x96 apparent pixel area, NOT near/far meters'
    measured['model_capability']=gate.capability
    measured['model_is_calibrated']=calibration is not None
    measured['warning']=('Validation-calibrated thresholds; independent test still required.'
        if calibration is not None else
        'Thresholds provisional. Calibrate on validation split; test only once.')
    performance={'execution_mode':'recorded images (NOT live USB camera)',
                 'utc':datetime.now(timezone.utc).isoformat(), 'model_sha256':hash_file(weight),
                 'model_path':str(weight.resolve()), 'machine':platform.platform(),
                 'python_version':platform.python_version(),
                 'images':len(gt_rows), 'elapsed_wall_seconds':duration,
                 'pipeline_throughput_fps':len(gt_rows)/duration if duration>0 else None,
                 'inference_tracking_ms':{'mean':1000*mean(latencies),
                   'p50':1000*percentile(latencies,50),
                   'p95':1000*percentile(latencies,95),
                   'p99':1000*percentile(latencies,99),
                   'max':1000*max(latencies)},
                 'image_read_mean_ms':1000*mean(io_time),
                 'dropped_frame_count':None,
                 'hardware_field_validated':False}
    out.mkdir(parents=True,exist_ok=True)
    for filename, rows in [('gt_test.jsonl',gt_frames),('pred_test.jsonl',predictions),('dual_scenes.jsonl',scenes)]:
        path=out/filename
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    json_dump_atomic(measured,out/'detection_tracking_metrics.json')
    json_dump_atomic(performance,out/'performance.json')
    return {'status':'OFFLINE_MEASURED', 'metrics':measured, 'performance':performance,
            'classification_counts':dict(classification_counts), 'report_dir':str(out.resolve())}
