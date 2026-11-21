"""TRUE recorded-media test: existing YOLO -> production LiveSessionManager -> LiveRecorder -> isolated SQLite.

Never uses USB, RTSP or synthetic model predictions. All video rights must be confirmed.
Does not touch the user's operational database, and does not claim GT metrics without GT.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda:fh.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def shorten_video(source:Path, output:Path, max_frames:int) -> tuple[int,float]:
    import cv2
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():raise RuntimeError('Recorded file cannot be decoded')
    writer=None
    count=0
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps<=0 or fps>240: fps=20.0
        for codec in ('MJPG','XVID'):
            # Read and rewind one sample to derive correct output dimensions.
            if count==0:
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Video contains no readable frames')
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            h,w=frame.shape[:2]
            writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*codec),fps,(w,h))
            if writer.isOpened():break
            writer.release()
        if writer is None or not writer.isOpened():raise RuntimeError('No video encoder supported for bounded recorded test')
        while count<max_frames:
            ok,frame=cap.read()
            if not ok:break
            if frame.ndim!=3 or frame.shape[2]!=3:raise RuntimeError('Invalid video frame')
            writer.write(frame)
            count+=1
        if count==0:raise RuntimeError('No frames encoded')
        return count,fps
    finally:
        cap.release()
        if writer is not None:writer.release()


def pct(values:list[float],p:float):
    if not values:return None
    a=sorted(values); i=(len(a)-1)*p; j=int(i); k=min(j+1,len(a)-1)
    return round(a[j]+(a[k]-a[j])*(i-j),3)


def run_video(video:Path,model_path:Path,max_frames:int,timeout:int,gt_path:Path|None,predictions_file:Path):
    from sqlalchemy import select, func
    from app.skyguard_persistence.database import Database
    from app.skyguard_persistence.models import Frame, Detection, Track, Session, PerformanceSample
    from app.skyguard_persistence.live_capture import LiveRecorder
    from app.detection.live_session import LiveSessionManager
    from app.detection.yolo import YOLODetectionEngine
    from app.video.source import VideoSourceConfig
    from .scientific import score_ground_truth
    import cv2

    if max_frames<2 or max_frames>500:raise ValueError('max_frames must be 2..500')
    if timeout<10 or timeout>1800:raise ValueError('timeout must be 10..1800 seconds')
    if not video.is_file() or video.stat().st_size==0:raise FileNotFoundError(video)
    if not model_path.is_file():raise FileNotFoundError(model_path)
    original_hash=sha256(video); model_hash=sha256(model_path)
    with tempfile.TemporaryDirectory(prefix='skyguard_offline_e2e_') as tmp:
        tmp=Path(tmp)
        bounded=tmp/'recorded_sample.avi'
        total,source_fps=shorten_video(video,bounded,max_frames)
        db_file=tmp/'isolated.sqlite3'
        db=Database(f'sqlite:///{db_file.as_posix()}')
        db.create_schema_for_tests_only()
        recorder=LiveRecorder(db,queue_capacity=2048)
        mgr=LiveSessionManager()
        model=YOLODetectionEngine(model_path=str(model_path),confidence_threshold=0.35)
        names=getattr(model.model,'names',{})
        class_list=sorted(str(x).lower() for x in (names.values() if isinstance(names,dict) else names))
        three_class=('drone' in class_list and 'bird' in class_list and
                     bool(set(class_list)&{'other_airborne','other','unknown'}))
        recorder.start(mgr)
        t0=time.perf_counter()
        error=None
        try:
            mgr.start(VideoSourceConfig('file',str(bounded)),model)
            thread=mgr._thread
            if thread is None:raise RuntimeError('No live worker was started')
            thread.join(timeout=timeout)
            if thread.is_alive():
                mgr.stop()
                thread.join(timeout=5)
                raise TimeoutError('Recorded video exceeded bounded processing timeout')
            snap=mgr.status()
            if snap.state=='error':raise RuntimeError('Production worker failed: '+str(snap.error))
            if snap.state!='stopped':raise RuntimeError(f'Unexpected worker final state {snap.state}')
            if not recorder.drain(timeout=20):raise RuntimeError('Recorder queue did not drain')
        finally:
            recorder.stop()
        elapsed=time.perf_counter()-t0
        health=recorder.health()
        live_id=mgr.status().session_id
        # Query the actual persisted records via SQLAlchemy after closing the writer.
        with db.sessionmaker() as s:
            found=s.scalars(select(Session).where(Session.dataset_ref==f'skyguard-live:{live_id}')).all()
            if len(found)!=1:raise RuntimeError('Expected one persisted source session')
            sid=found[0].id
            frames=s.scalars(select(Frame).where(Frame.session_id==sid).order_by(Frame.frame_index)).all()
            dets=s.execute(select(Detection,Frame.frame_index).join(Frame,Frame.id==Detection.frame_id)
                           .where(Frame.session_id==sid)).all()
            tracks=s.scalars(select(Track).where(Track.session_id==sid)).all()
            samples=s.scalars(select(PerformanceSample).where(PerformanceSample.session_id==sid)).all()
            observed_indices=[f.frame_index for f in frames]
            if observed_indices!=list(range(total)):
                raise RuntimeError(f'Persisted frames mismatch: {len(observed_indices)} of {total}; indices not monotonic/gap-free')
            if health['lost_frames'] or health['errors'] or health['queued_frames']:
                raise RuntimeError(f"Recorder health degraded: {health['lost_frames']} lost; {health['errors']} errors")
            if any(d.classification=='UNKNOWN' and d.track_id is not None for d,_ in dets):
                raise RuntimeError('UNKNOWN was persisted with tracking ID')
            if any(d.classification=='UNKNOWN' for d,_ in dets) and any(t.classification=='UNKNOWN' for t in tracks):
                raise RuntimeError('UNKNOWN received a persisted track')
            detections=Counter(d.classification for d,_ in dets)
            # Produce true per-frame predictions from persisted outputs, not guessed GT.
            pred=[{'sequence_id':'clip','frame_index':i,'objects':[]} for i in range(total)]
            for d,idx in dets:
                pred[idx]['objects'].append({'class':d.classification,
                                            'bbox':[d.x1,d.y1,d.x2,d.y2],
                                            'track_id':next((t.external_track_id for t in tracks if t.id==d.track_id),None)})
            timings=[float(p.latency_ms) for p in samples if p.latency_ms is not None]
        # Reopen the DB after disposing the writer; prove durable access.
        db.dispose()
        reopened=Database(f'sqlite:///{db_file.as_posix()}')
        with reopened.sessionmaker() as s:
            after=s.scalar(select(func.count(Frame.id)).where(Frame.session_id==sid))
            if after!=total:raise RuntimeError('Data lost on database reopen')
        reopened.dispose()
        metrics=None
        if gt_path is not None:
            annotated=json.loads(gt_path.read_text(encoding='utf8'))
            metrics=score_ground_truth(annotated,pred)
        predictions_file.parent.mkdir(parents=True,exist_ok=True)
        predictions_file.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in pred),encoding='utf8')
        return {
            'result':'PASS_RECORDED_E2E_NO_FIELD_CLAIM',
            'predictions_sha256':sha256(predictions_file),
            'predictions_file':str(predictions_file),
            'source_video_sha256':original_hash,'model_sha256':model_hash,
            'model_classes':class_list,'tri_class_model':three_class,
            'source_fps_metadata':source_fps, 'frames_in_recorded_clip':total,
            'frames_persisted':len(observed_indices),'detections':dict(detections),
            'tracks_persisted':len(tracks),'recorder':health,
            'latency_ms':{'p50':pct(timings,.5),'p95':pct(timings,.95),'p99':pct(timings,.99)},
            'measured_end_to_end_fps_including_model_start':round(total/elapsed,3),
            'elapsed_wall_s':round(elapsed,3),
            'timestamp_semantics':'host_publication_utc_not_camera_sensor_capture',
            'persistence_after_reopen':True,
            'ground_truth_metrics':metrics,
            'classification_acceptance':('GT_EVALUATION_AVAILABLE_REVIEW_METRICS' if metrics and three_class
                                         else 'BLOCKED_NO_SOURCE_DISJOINT_VERIFIED_GT_OR_TRI_CLASS_MODEL'),
            'hota':'NOT_COMPUTED',
            'field_validation':'NOT_PERFORMED',
            'data_rights':'USER_ATTESTED_NOT_INDEPENDENTLY_AUDITED',
        }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--max-frames',type=int,default=80)
    p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--confirm-rights',action='store_true')
    p.add_argument('--ground-truth',type=Path)
    a=p.parse_args()
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'result':'BLOCKED','field_validation':'NOT_PERFORMED'}
    rc=1
    try:
        if not a.confirm_rights:raise PermissionError('Explicit right to process video not confirmed')
        report=run_video(a.video.resolve(),a.model.resolve(),a.max_frames,a.timeout,a.ground_truth,
                         a.report.with_name("recorded_predictions.jsonl"))
        rc=0
    except Exception as exc:
        report['error_kind']=type(exc).__name__
        report['error']=str(exc)[:600]
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'result':report['result'],'report':str(a.report),
                      'error_kind':report.get('error_kind')},ensure_ascii=False))
    raise SystemExit(rc)

if __name__=='__main__':main()
