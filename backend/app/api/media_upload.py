"""Original SkyGuard app media adapter. Localhost, API-key protected by app.main.

Runs real checkpoint inference, never constructs synthetic labels or ranges.
The composed API persists job metadata in the existing media jobs table.
"""
from __future__ import annotations

import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from skyguard_fix.cli import draw, evaluate_file
from skyguard_fix.core import UltralyticsPredictor, multiscale

router = APIRouter(prefix='/api/media', tags=['SkyGuard original dashboard media'])
_ROOT = Path(__file__).resolve().parents[2]
_REPORT = _ROOT / 'reports' / 'operator_media'
_MODEL_DEFAULT = _ROOT / 'models' / 'skyguard-drone-bird-t4-v1' / 'best.pt'
_MAX_BYTES = 80 * 1024 * 1024
_jobs: dict[str, dict] = {}
_lock = threading.RLock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='skyguard-media')
_predictor = None
_predictor_path = None
_predict_lock = threading.RLock()
_browser_index = 0
_browser_running = False


def _model_path() -> Path:
    return Path(os.environ.get('SKYGUARD_MODEL_PATH') or _MODEL_DEFAULT).resolve()


def _require_model_file() -> Path:
    path = _model_path()
    if not path.is_file():
        raise HTTPException(503, 'Configured model weights are missing')
    return path


def _model():
    global _predictor, _predictor_path
    path = _require_model_file()
    with _predict_lock:
        if _predictor is None or _predictor_path != path:
            _predictor = UltralyticsPredictor(path)
            _predictor_path = path
        return _predictor


def _file_kind(data: bytes | bytearray, declared: str) -> tuple[str, str]:
    t = declared.upper()
    if t == 'IMAGE' and data.startswith(b'\xff\xd8\xff'):
        return 'IMAGE', '.jpg'
    if t == 'IMAGE' and data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'IMAGE', '.png'
    if t == 'VIDEO' and len(data) >= 12 and data[4:8] == b'ftyp':
        return 'VIDEO', '.mp4'
    raise HTTPException(415, 'Only valid JPG, PNG, or MP4 input is supported')


def _summary(detections_jsonl: Path):
    count = Counter()
    review_count = 0
    times = []
    for line in detections_jsonl.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        times.append(float(row.get('inference_ms', 0)))
        for d in row.get('detections', []):
            if d.get('verdict_status') == 'CONFLICT_REVIEW':
                review_count += 1
                continue
            cls = str(d.get('label', '')).upper()
            count[cls] += 1
            # Raw independent detections DO NOT establish identity persistence.
            # No synthetic IDs or trajectories are created for this uploaded clip.
    return {
        'detections_per_frame_total': dict(count),
        'conflict_review_total': review_count,
        'unique_confirmed_tracks': {},
        'mean_inference_pipeline_ms': round(sum(times)/len(times), 2) if times else None,
        'track_paths_px': [],
        'scientific_accuracy': 'NOT_MEASURED_WITH_GROUND_TRUTH',
        'note': 'Per-frame inference; tracking IDs require verified tracking adapter. No GPS or range inferred.',
    }


def _update_job(job_store, jobid: str, **values):
    if job_store is not None:
        job_store.update(jobid, **values)
    else:
        with _lock:
            _jobs[jobid].update(values)


def _get_job(request: Request, jid: str):
    job_store = getattr(request.app.state, 'skyguard_media_jobs', None)
    if job_store is not None:
        return job_store.get(jid)
    with _lock:
        job = _jobs.get(jid)
        return dict(job) if job else None


def _worker(jobid: str, source: Path, kind: str, job_store):
    try:
        _update_job(job_store, jobid, status='RUNNING')
        predictor = _model()
        def report_progress(frames_processed: int):
            if frames_processed == 1 or frames_processed % 5 == 0:
                _update_job(job_store, jobid, frames_processed=frames_processed)

        report = evaluate_file(
            source, source.parent / 'inference', predictor,
            max_frames=0, include_tiles=True, video_webm=True, on_progress=report_progress,
            inference_lock=_predict_lock,
        )
        folder = source.parent / 'inference'
        summary = _summary(folder / 'detections.jsonl')
        preview_path = folder / report['prediction_file']
        if not preview_path.is_file() or preview_path.stat().st_size == 0:
            raise RuntimeError('Annotated output was not produced')
        if kind == 'IMAGE':
            image = cv2.imread(str(preview_path))
        else:
            cap = cv2.VideoCapture(str(preview_path))
            try:
                success, image = cap.read()
                if not success:
                    raise RuntimeError('Annotated video cannot be decoded')
            finally:
                cap.release()
        if image is None:
            raise RuntimeError('Preview cannot be decoded')
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            raise RuntimeError('Preview JPEG encoding failed')
        (source.parent / 'preview.jpg').write_bytes(buf.tobytes())
        summary['prediction_url'] = f'/api/media/jobs/{jobid}/prediction'
        summary['model_classes'] = report['model_labels']
        summary['unknown_trained'] = bool(report['v2_has_unknown'])
        summary['latency_ms_p50'] = report['latency_ms_p50']
        summary['latency_ms_p95'] = report['latency_ms_p95']
        (source.parent / 'result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        _update_job(job_store, jobid, status='DONE', frames_processed=report['frames_processed'], summary=summary)
    except Exception as e:
        _update_job(job_store, jobid, status='FAILED', error=f'{type(e).__name__}: {e}'[:512])


@router.get('/health')
def media_health():
    path = _model_path()
    return {'status': 'ok', 'model_found': path.is_file(), 'model_path': str(path), 'upload_max_bytes': _MAX_BYTES}


@router.post('/upload', status_code=202)
async def upload(request: Request):
    declared = request.headers.get('x-skyguard-media-type', '')
    if declared.upper() not in ('IMAGE', 'VIDEO'):
        raise HTTPException(415, 'Missing IMAGE or VIDEO type header')
    length = request.headers.get('content-length')
    if length and int(length) > _MAX_BYTES:
        raise HTTPException(413, 'Upload exceeds 80 MiB')
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > _MAX_BYTES:
            raise HTTPException(413, 'Upload exceeds 80 MiB')
        data.extend(chunk)
    if not data:
        raise HTTPException(413, 'Empty upload')
    kind, ext = _file_kind(data, declared)
    _require_model_file()
    jid = uuid.uuid4().hex
    folder = _REPORT / jid
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / ('source'+ext)
    source.write_bytes(data)
    job_store = getattr(request.app.state, 'skyguard_media_jobs', None)
    sha256 = hashlib.sha256(data).hexdigest()
    if job_store is not None:
        job_store.create(jid, kind, sha256, f'{jid}/{source.name}')
    else:
        with _lock:
            _jobs[jid] = {'id': jid, 'status': 'QUEUED', 'kind': kind, 'frames_processed': 0,
                          'session_id': None, 'error': None, 'input_sha256': sha256}
    _executor.submit(_worker, jid, source, kind, job_store)
    return {'job_id': jid, 'status': 'QUEUED'}


@router.get('/jobs/{jid}')
def job_status(jid: str, request: Request):
    job = _get_job(request, jid)
    if job is None:
        raise HTTPException(404, 'Unknown job ID')
    preview = _REPORT / jid / 'preview.jpg'
    if job['status'] == 'DONE' and preview.is_file():
        job['summary'] = {
            **job['summary'],
            'preview_jpeg_base64': base64.b64encode(preview.read_bytes()).decode('ascii'),
        }
    return job


@router.get('/jobs/{jid}/prediction')
def prediction(jid: str, request: Request):
    job = _get_job(request, jid)
    if not job or job['status'] != 'DONE':
        raise HTTPException(404, 'Prediction not ready')
    folder = _REPORT / jid / 'inference'
    if job['kind'] == 'IMAGE':
        path, media_type = folder / 'prediction.png', 'image/png'
    elif (folder / 'prediction.webm').is_file() and not (folder / 'prediction.mp4').is_file():
        path, media_type = folder / 'prediction.webm', 'video/webm'
    else:
        path, media_type = folder / 'prediction.mp4', 'video/mp4'
    if not path.is_file():
        raise HTTPException(404, 'Prediction artifact missing')
    return FileResponse(path, media_type=media_type, headers={'Cache-Control':'no-store'})


@router.post('/browser/start')
def browser_start():
    global _browser_running, _browser_index
    _model()
    with _lock:
        _browser_index = 0
        _browser_running = True
    return {'status':'running', 'real_model':True}


@router.post('/browser/stop')
def browser_stop():
    global _browser_running
    with _lock:
        _browser_running = False
    return {'status':'stopped'}


def _analyze_browser_frame(frame):
    with _predict_lock:
        model = _model()
        return multiscale(frame, model, model.names, include_tiles=False)


@router.post('/browser/frame')
async def browser_frame(request: Request):
    global _browser_index
    with _lock:
        if not _browser_running:
            raise HTTPException(409, 'Start browser camera first')
        frame_idx = _browser_index
        _browser_index += 1
    data = await request.body()
    if len(data)>4*1024*1024 or not data.startswith(b'\xff\xd8\xff'):
        raise HTTPException(415, 'Expected JPEG frame <= 4 MiB')
    import numpy as np
    frame = cv2.imdecode(np.frombuffer(data, dtype='uint8'), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, 'Invalid JPEG frame')
    h,w = frame.shape[:2]
    if not (32<=w<=4096 and 32<=h<=4096):
        raise HTTPException(422, 'Camera dimensions unsupported')
    start = time.perf_counter()
    dets = await run_in_threadpool(_analyze_browser_frame, frame)
    result = [
        {
            'classification': d.label if d.verdict_status != 'CONFLICT_REVIEW' else None,
            'verdict_status': d.verdict_status,
            'track_id': None,
            'confidence': d.confidence,
            'bbox': list(d.xyxy),
            'range_m': None,
            'latitude': None,
            'longitude': None,
        }
        for d in dets
    ]
    frame = draw(frame, dets)
    ok,jpg=cv2.imencode('.jpg',frame)
    if not ok:
        raise HTTPException(500, 'Encoding failed')
    return {'frame_index':frame_idx,'session_id':'browser-local', 'detections':result,
            'pixel_trajectories':[], 'confirmed_unique_id_count':{},
            'latency_ms':round((time.perf_counter()-start)*1000,2), 'range_m':None,
            'preview_jpeg_base64':base64.b64encode(jpg.tobytes()).decode('ascii'),
            'tracking_status':'NOT_ESTABLISHED_FOR_BROWSER_FRAMES'}
