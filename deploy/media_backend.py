"""SkyGuard FUSION authenticated media processing.

Actual model inference via existing YOLODetectionEngine, ObjectTracker and Store.
No artificial detections, no inferred GPS/range, no UNKNOWN tracking.
Single backend worker and single media processing worker in this deployment profile.
"""
from __future__ import annotations

import asyncio
import base64
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import threading
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import text, bindparam, JSON

from app.skyguard_persistence.live_integrated import app
from app.skyguard_persistence.database import Database
from app.skyguard_persistence.schemas import CameraCreate, SessionCreate, ObjectInput, FrameIngest
from app.skyguard_persistence.service import Store
from app.detection.yolo import YOLODetectionEngine
from app.detection.tracker import ObjectTracker
from app.detection.engine import DetectionValidator

ROOT = Path(os.environ.get('SKYGUARD_MEDIA_ROOT', '/opt/skyguard/backend/media')).resolve()
MODEL = Path(os.environ.get('SKYGUARD_MODEL_PATH', '/opt/skyguard/backend/models/skyguard-drone-bird-t4-v1/best.pt')).resolve()
EXPECTED_MODEL_SHA = os.environ.get('SKYGUARD_EXPECTED_MODEL_SHA256','').lower()
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 40 * 1024 * 1024
MAX_FRAMES = int(os.environ.get('SKYGUARD_MEDIA_MAX_FRAMES','120'))
MAX_FRAMES = min(240,max(2,MAX_FRAMES))
security = HTTPBearer(auto_error=False)
router = APIRouter(prefix='/api/media',tags=['Authenticated operator media inputs'])
db = Database()
store = Store(db)
pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='skyguard-media')
browser_lock = threading.Lock()
browser_state = None

def utcnow(): return datetime.now(timezone.utc)

def auth(credentials:HTTPAuthorizationCredentials|None=Depends(security)):
    wanted=os.environ.get('SKYGUARD_DB_API_KEY','')
    if not wanted or credentials is None or not secrets.compare_digest(credentials.credentials,wanted):
        raise HTTPException(401,'Authorized backend request required')

def _verify_model_file():
    if not MODEL.is_file(): raise HTTPException(503,'MODEL_MISSING: trained SkyGuard checkpoint required')
    if not re.fullmatch(r'[a-f0-9]{64}',EXPECTED_MODEL_SHA or ''):
        raise HTTPException(503,'SKYGUARD_EXPECTED_MODEL_SHA256 must contain independently verified model digest')
    h=sha256()
    with MODEL.open('rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b''): h.update(chunk)
    if not secrets.compare_digest(h.hexdigest(),EXPECTED_MODEL_SHA):
        raise HTTPException(503,'MODEL_SHA256_MISMATCH; refusing unverified model')
    return True

def _model():
    _verify_model_file()
    return YOLODetectionEngine(model_path=str(MODEL),confidence_threshold=0.35)

def _source_type(data:bytes,kind:str)->str:
    if kind=='IMAGE':
        if data[:3]==b'\xff\xd8\xff':return '.jpg'
        if data[:8]==b'\x89PNG\r\n\x1a\n':return '.png'
    if kind=='VIDEO':
        if len(data)>12 and data[4:8]==b'ftyp':return '.mp4'
        if data[:4]==b'RIFF' and data[8:12]==b'AVI ':return '.avi'
        if data[:4]==b'\x1a\x45\xdf\xa3':return '.webm'
    raise HTTPException(415,'Only real JPEG/PNG images or MP4/AVI/WebM video bytes supported')

def _image(data:bytes):
    try:
        with Image.open(BytesIO(data)) as im:
            if im.width*im.height>24_000_000 or im.width<16 or im.height<16:
                raise ValueError('Image pixel dimensions unsupported')
            img=ImageOps.exif_transpose(im).convert('RGB')
            rgb=np.asarray(img)
            return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError):
        raise HTTPException(415,'Invalid or excessively large image')

def _detector_frame(detector, tracker, frame, index:int, session_id:str, src_sha:str, make_preview:bool=True):
    start=perf_counter()
    ds=detector.detect(frame)
    DetectionValidator.validate(frame,ds)
    eligible=[d for d in ds if str(d.object_class).lower() in ('drone','bird')]
    result=tracker.update(frame_index=index,detections=eligible) if tracker is not None else []
    observed=list(getattr(result,'confirmed',result)) if result is not None else []
    matches={}
    for t in observed:
        d=getattr(t,'detection',None)
        if d is not None and str(d.object_class).lower() in ('drone','bird') and getattr(t,'state','')=='confirmed':
            matches[id(d)]=str(t.track_id)
    objects=[]
    for d in ds:
        cls=str(d.object_class).lower()
        label=cls.upper() if cls in ('drone','bird') else 'UNKNOWN'
        b=d.bbox
        track_id=matches.get(id(d)) if label!='UNKNOWN' else None
        objects.append(ObjectInput(classification=label,
            bbox=(float(b.x),float(b.y),float(b.x+b.width),float(b.y+b.height)),
            confidence=float(d.confidence),track_id=track_id))
    now=utcnow()
    store.ingest_frame(FrameIngest(session_id=session_id,frame_index=index,captured_at=now,objects=objects),actor='backend_media_ai')
    latency_ms=round((perf_counter()-start)*1000,2)
    preview=None
    if make_preview:
        annotated=frame.copy()
        for ob in objects:
            x1,y1,x2,y2=(int(x) for x in ob.bbox)
            color=(0,175,255) if ob.classification=='UNKNOWN' else ((80,215,80) if ob.classification=='DRONE' else (240,175,70))
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            label=ob.classification+(f' #{ob.track_id}' if ob.track_id else ' UNTRACKED')
            cv2.putText(annotated,label,(max(0,x1),max(15,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.48,color,1)
        h,w=annotated.shape[:2]
        if max(h,w)>1280:
            scale=1280/max(h,w)
            annotated=cv2.resize(annotated,(int(w*scale),int(h*scale)))
        ok,enc=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,70])
        if ok and len(enc)<1_400_000: preview=base64.b64encode(enc.tobytes()).decode('ascii')
    return objects,latency_ms,preview

def _camera_session(jobid:str,kind:str,origin:str):
    camera=store.create_camera(CameraCreate(name='SkyGuard '+origin,source_kind='FILE' if kind in ('IMAGE','VIDEO') else 'USB',external_ref='media:'+jobid),actor='backend_media_ai')
    ses=store.create_session(SessionCreate(camera_id=camera['id'],source_kind=kind,validation_level='SOFTWARE_ONLY',dataset_ref='media:'+jobid),actor='backend_media_ai')
    return ses['id']

def _job_update(jobid,status,**updates):
    allowed={'session_id','frames_processed','summary','error'}
    if set(updates)-allowed: raise RuntimeError('Unexpected job field')
    clauses=['status=:state','updated_at=:updated_at']
    args={'state':status,'updated_at':utcnow(),'id':jobid}
    for key,value in updates.items():
        clauses.append(key+'=:'+key)
        args[key]=value
    with db.engine.begin() as con:
        stmt=text('UPDATE sg_media_jobs SET '+','.join(clauses)+' WHERE id=:id')
        if 'summary' in updates:stmt=stmt.bindparams(bindparam('summary',type_=JSON()))
        con.execute(stmt,args)

def _job_read(jobid):
    with db.engine.connect() as con:
        r=con.execute(text('SELECT id,kind,status,source_sha256,session_id,frames_processed,summary,error,created_at,updated_at FROM sg_media_jobs WHERE id=:id'),{'id':jobid}).mappings().one_or_none()
    if not r: raise HTTPException(404,'Job not found')
    result=dict(r)
    for key in ('created_at','updated_at'):
        if isinstance(result[key],datetime):result[key]=result[key].isoformat()
    if isinstance(result['summary'],str):result['summary']=json.loads(result['summary'])
    return result

def _process(jobid,path,kind,src_hash):
    _job_update(jobid,'RUNNING')
    session_id=None
    try:
        detector=_model()
        tracker=ObjectTracker()
        tracker.reset()
        session_id=_camera_session(jobid,kind,'upload')
        _job_update(jobid,'RUNNING',session_id=session_id)
        n=0; counts=Counter(); unique=defaultdict(set); timings=[]; image=None
        if kind=='IMAGE':
            img=_image(path.read_bytes())
            objects,ms,image=_detector_frame(detector,None,img,0,session_id,src_hash)
            n=1;timings.append(ms)
            for ob in objects: counts[ob.classification]+=1
        else:
            cap=cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened(): raise ValueError('Video cannot be decoded on server')
                while n<MAX_FRAMES:
                    ok,frame=cap.read()
                    if not ok:break
                    if not isinstance(frame,np.ndarray) or frame.ndim!=3:raise ValueError('Invalid decoded frame')
                    if frame.shape[0]*frame.shape[1]>24_000_000:raise ValueError('Frame is too large')
                    objects,ms,preview=_detector_frame(detector,tracker,frame,n,session_id,src_hash,make_preview=(n%8==0))
                    timings.append(ms)
                    if preview: image=preview
                    for ob in objects:
                        counts[ob.classification]+=1
                        if ob.track_id:unique[ob.classification].add(ob.track_id)
                    n+=1
                    if n%10==0: _job_update(jobid,'RUNNING',frames_processed=n)
            finally:cap.release()
            if n==0:raise ValueError('No video frames decoded')
        store.close_session(session_id,actor='backend_media_ai')
        tracks=store.list_tracks(session_id,limit=200) if kind=='VIDEO' else []
        track_paths=[]
        for t in tracks:
            hist=store.track_history(t['id'],limit=MAX_FRAMES)
            points=[]
            for h in hist:
                d=h['detection']
                points.append({'frame':h['observation']['frame_index'], 'x_px':round((d['x1']+d['x2'])/2,2),'y_px':round((d['y1']+d['y2'])/2,2)})
            if points:track_paths.append({'track_id':t['external_track_id'],'classification':t['classification'],'points':points})
        summary={'source_sha256':src_hash,'frames_processed':n,'detections_per_frame_total':dict(counts),
                 'unique_confirmed_tracks':{k:len(v) for k,v in unique.items()},
                 'mean_inference_pipeline_ms':round(sum(timings)/len(timings),2) if timings else None,
                 'track_paths_px':track_paths[:200],'preview_jpeg_base64':image,
                 'range_m':None,'gps':None,'scientific_accuracy':'NOT_MEASURED_NO_GROUND_TRUTH',
                 'timestamp_semantics':'backend_processing_time_not_original_sensor_time',
                 'note':'Photo: no trajectory. Video trajectories are pixels, not meters. UNKNOWN is never tracked.'}
        _job_update(jobid,'DONE',frames_processed=n,summary=summary)
    except Exception as exc:
        if session_id:
            try:store.close_session(session_id,actor='backend_media_ai')
            except Exception:pass
        _job_update(jobid,'FAILED',error=(type(exc).__name__+': '+str(exc))[:480])

@router.get('/health',dependencies=[Depends(auth)])
def health():
    return {'api':'online','model_present':MODEL.is_file(),
            'model_sha256_configured':bool(re.fullmatch(r'[a-f0-9]{64}',EXPECTED_MODEL_SHA or '')),
            'postgresql':db.engine.dialect.name=='postgresql','distance':'UNVERIFIED_WITHOUT_SPATIAL_CALIBRATION'}

@router.post('/upload',dependencies=[Depends(auth)],status_code=202)
async def upload(request:Request):
    kind=request.headers.get('x-skyguard-media-type','').upper()
    if kind not in ('IMAGE','VIDEO'):raise HTTPException(422,'IMAGE or VIDEO media type required')
    size_max=MAX_IMAGE_BYTES if kind=='IMAGE' else MAX_VIDEO_BYTES
    data=bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data)>size_max:raise HTTPException(413,'Media file exceeds configured size cap')
    if not data:raise HTTPException(400,'Empty file')
    ext=_source_type(data,kind)
    # Refuse unusable checkpoint before creating an apparently successful job.
    _verify_model_file()
    jobid=str(uuid4()); filename=jobid+ext
    ROOT.mkdir(parents=True,exist_ok=True)
    dest=ROOT/filename
    dest.write_bytes(data)
    now=utcnow()
    with db.engine.begin() as con:
        con.execute(text('INSERT INTO sg_media_jobs (id,kind,status,source_sha256,source_file,session_id,frames_processed,summary,error,created_at,updated_at) '
                         'VALUES (:id,:kind,:status,:sha,:file,NULL,0,:summary,NULL,:created,:updated)').bindparams(bindparam('summary',type_=JSON())),
                    {'id':jobid,'kind':kind,'status':'QUEUED','sha':sha256(data).hexdigest(),
                     'file':filename,'summary':{},'created':now,'updated':now})
    pool.submit(_process,jobid,dest,kind,sha256(data).hexdigest())
    return {'job_id':jobid,'status':'QUEUED','source_sha256':sha256(data).hexdigest()}

@router.get('/jobs/{job_id}',dependencies=[Depends(auth)])
def job(job_id:str):
    if not re.fullmatch(r'[0-9a-f-]{36}',job_id):raise HTTPException(400,'Invalid job ID')
    return _job_read(job_id)

class BrowserState:
    def __init__(self,model,session_id):
        self.model=model
        self.session_id=session_id
        self.tracker=ObjectTracker()
        self.tracker.reset()
        self.frame=0
        self.known=defaultdict(set)
        self.history=defaultdict(list)

@router.post('/browser/start',dependencies=[Depends(auth)])
def browser_start():
    global browser_state
    with browser_lock:
        if browser_state is not None:raise HTTPException(409,'Browser camera already running')
        detector=_model()
        sid=_camera_session(uuid4().hex,'LIVE','browser_camera')
        browser_state=BrowserState(detector,sid)
        return {'status':'RUNNING','session_id':sid,'fps_limit':2,'distance_m':None}

@router.post('/browser/frame',dependencies=[Depends(auth)])
async def browser_frame(request:Request):
    global browser_state
    data=bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data)>2_000_000:raise HTTPException(413,'Browser JPEG too large')
    if _source_type(data,'IMAGE')!='.jpg':raise HTTPException(415,'JPEG camera frame required')
    frame=_image(data)
    if not browser_lock.acquire(blocking=False):raise HTTPException(429,'Camera processing busy; retry later')
    try:
        st=browser_state
        if st is None:raise HTTPException(409,'Start camera first')
        if st.frame>=3000:raise HTTPException(409,'Camera session limit reached; stop and restart')
        objects,ms,preview=await asyncio.to_thread(_detector_frame,st.model,st.tracker,frame,st.frame,st.session_id,'browser')
        for ob in objects:
            if ob.track_id:
                st.known[ob.classification].add(ob.track_id)
                pts=st.history[ob.track_id]
                pts.append({'frame':st.frame,'x_px':round((ob.bbox[0]+ob.bbox[2])/2,2),
                            'y_px':round((ob.bbox[1]+ob.bbox[3])/2,2)})
                del pts[:-80]
        st.frame+=1
        return {'session_id':st.session_id,'frame_index':st.frame-1,
                'detections':[o.model_dump() for o in objects],
                'confirmed_unique_id_count':{x:len(v) for x,v in st.known.items()},
                'active_confirmed_objects':dict(Counter(o.classification for o in objects if o.track_id)),
                'pixel_trajectories':[{ 'track_id':k,'points':p} for k,p in st.history.items()],
                'latency_ms':ms,'preview_jpeg_base64':preview,'range_m':None,
                'gps':None,'source':'REAL_BROWSER_CAMERA_BACKEND_AI_SOFTWARE_ONLY'}
    finally:browser_lock.release()

@router.post('/browser/stop',dependencies=[Depends(auth)])
def browser_stop():
    global browser_state
    with browser_lock:
        if browser_state is not None:
            sid=browser_state.session_id
            store.close_session(sid,actor='backend_media_ai')
            browser_state=None
            return {'status':'STOPPED','session_id':sid}
        return {'status':'IDLE'}

@app.on_event('startup')
def media_startup():
    # Previous incomplete jobs remain auditable rather than faking completion.
    with db.engine.begin() as con:
        con.execute(text("UPDATE sg_media_jobs SET status='INTERRUPTED',error='Backend restarted before completion',updated_at=:now WHERE status IN ('RUNNING','QUEUED')"),{'now':utcnow()})

@app.on_event('shutdown')
def media_shutdown():
    pool.shutdown(wait=False,cancel_futures=True)
    db.dispose()

app.include_router(router)
