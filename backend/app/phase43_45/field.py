"""Phase 45 operator-initiated REAL USB camera recording/performance protocol.

Not run in phases 43-44. Never claims physical drone was observed without
independent operator ground-truth annotation and reference measurements.
"""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4
import json
import platform

from .benchmark import percentile, _model_predictions, _names
from .inventory import hash_file,json_dump_atomic
from .perception import DecisionGate, TargetOnlyTracker
from .maps import dual_scene
from .acceptance import field_template


def usb_camera_trial(*, camera_index: int, weights: str | Path,
                     destination: str | Path, max_frames: int=300,
                     physical_confirmed: bool=False, imgsz: int=640) -> dict:
    if not physical_confirmed:
        raise PermissionError('Phase 45 only: explicit --physical-confirm required')
    if not 0 <= camera_index <= 32 or max_frames <= 0 or imgsz < 32:
        raise ValueError('Invalid camera test parameters')
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install opencv-python and ultralytics in backend venv') from exc
    output=Path(destination)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Trial directory must be new/empty')
    output.mkdir(parents=True,exist_ok=True)
    weight=Path(weights)
    if not weight.is_file():
        raise FileNotFoundError(weight)
    model=YOLO(str(weight))
    gate=DecisionGate(_names(model))
    if not gate.capability['open_set_candidate']:
        raise ValueError('Physical open-set camera trial requires 3/4-class checkpoint')
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if platform.system()=='Windows' else cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('USB camera did not open; no field evidence generated')
    stream_id=str(uuid4())
    output_video=output/'raw_camera.avi'
    writer=None
    tracker=TargetOnlyTracker(min_hits=2)
    latencies=[]; read_times=[]; time_marks=[]
    processed=0
    t_start=perf_counter()
    try:
        with (output/'detections.jsonl').open('w',encoding='utf-8') as det_file, \
             (output/'dual_scenes.jsonl').open('w',encoding='utf-8') as map_file:
            for i in range(max_frames):
                read_start=perf_counter()
                ok,frame=cap.read()
                read_times.append(perf_counter()-read_start)
                if not ok or frame is None:
                    break
                now=perf_counter()
                time_marks.append(now)
                height,width=frame.shape[:2]
                if writer is None:
                    writer=cv2.VideoWriter(str(output_video),cv2.VideoWriter_fourcc(*'MJPG'),
                                           max(1.0,float(cap.get(cv2.CAP_PROP_FPS)) or 30.0),
                                           (width,height))
                    if not writer.isOpened():
                        raise RuntimeError('Video evidence writer failed to open')
                writer.write(frame)
                started=perf_counter()
                choices=[gate.decide(d) for d in _model_predictions(model,frame,low_conf=.10,imgsz=imgsz)]
                tracks=tracker.update(i,choices)
                latencies.append(perf_counter()-started)
                timestamp_ms=(now-t_start)*1000
                det_file.write(json.dumps({'session_id':stream_id,'frame_index':i,
                    'timestamp_ms':timestamp_ms,'detections':choices,
                    'tracks':tracks['confirmed']},ensure_ascii=False)+'\n')
                scene=dual_scene(stream_id,i,timestamp_ms,tracks['confirmed'],width)
                map_file.write(json.dumps(scene,ensure_ascii=False)+'\n')
                processed+=1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    duration=perf_counter()-t_start
    if processed==0:
        raise RuntimeError('No USB frames captured')
    report={'status':'RECORDED_PHYSICAL_INPUT_UNVERIFIED',
        'warning':'USB input capture is not evidence of a real DRONE or ground-truth accuracy',
        'trial_session_id':stream_id,'utc':datetime.now(timezone.utc).isoformat(),
        'operator_confirmed_usb':physical_confirmed,'usb_camera_index':camera_index,
        'host':platform.platform(), 'model_sha256':hash_file(weight),
        'frames_processed':processed,'wall_seconds':duration,
        'measured_pipeline_output_fps':processed/duration,
        'consecutive_input_arrival_mean_ms':1000*mean([b-a for a,b in zip(time_marks,time_marks[1:])])
             if len(time_marks)>1 else None,
        'camera_read_ms_p95':1000*percentile(read_times,95),
        'inference_tracking_ms_p50':1000*percentile(latencies,50),
        'inference_tracking_ms_p95':1000*percentile(latencies,95),
        'inference_tracking_ms_p99':1000*percentile(latencies,99),
        'dropped_frame_count':None,
        'raw_footage_sha256':hash_file(output_video),
        'independent_range_reference':None,'drone_present_gt':None,
        'field_success_claimed':False}
    json_dump_atomic(report,output/'usb_telemetry.json')
    template=field_template()
    template['trial_id']=stream_id
    template['evidence']['video_sha256']=report['raw_footage_sha256']
    template['measurements']['output_fps']=report['measured_pipeline_output_fps']
    template['measurements']['p50_latency_ms']=report['inference_tracking_ms_p50']
    template['measurements']['p95_latency_ms']=report['inference_tracking_ms_p95']
    template['measurements']['p99_latency_ms']=report['inference_tracking_ms_p99']
    json_dump_atomic(template,output/'field_annotation_PENDING.json')
    return report
