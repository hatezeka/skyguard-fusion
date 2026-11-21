"""Deterministic Halmstad MCOS -> YOLO11 4-class export.

Requires opencv-python + mcos-decoder. Streams one ZIP video at a time into
private temporary file; does NOT modify or embed media in the repository.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import math
import shutil

from .inventory import CLASSES, json_dump_atomic


def _bbox_to_yolo(bbox, width: int, height: int) -> tuple[float,float,float,float] | None:
    if bbox is None:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox]
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Malformed MCOS bbox {bbox!r}') from exc
    if not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0:
        raise ValueError('Invalid bbox')
    x1, y1 = max(0.,x), max(0.,y)
    x2, y2 = min(width,x+w), min(height,y+h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('BBox outside source image')
    return ((x1+x2)/(2*width), (y1+y2)/(2*height),
            (x2-x1)/width, (y2-y1)/height)


def export_from_archive(manifest: dict, destination: str | Path, *,
                        stride: int = 15, max_sequences: int = 0,
                        max_frames_per_video: int = 0) -> dict:
    if stride <= 0 or max_sequences < 0 or max_frames_per_video < 0:
        raise ValueError('stride > 0, limits >= 0 required')
    try:
        import cv2
        from mcos_decoder import load_groundtruth
    except ImportError as ex:
        raise RuntimeError('Install: pip install opencv-python mcos-decoder') from ex
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Output must be new/empty: {root}')
    records = manifest['records']
    if max_sequences:
        # Ensure early smoke includes all classes and modalities, then hash order.
        from hashlib import sha256
        # Round-robin buckets preserve class, split, and sensor diversity in smoke runs.
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in records:
            buckets[(r['split'], r['class'], r['modality'])].append(r)
        for bucket in buckets.values():
            bucket.sort(key=lambda r:sha256(r['sequence_id'].encode()).hexdigest())
        selected = []
        while len(selected) < max_sequences and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(selected) < max_sequences:
                    selected.append(buckets[key].pop(0))
        records = selected
    root.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    failures = []
    gt_path = root / 'ground_truth.jsonl'
    source_zip = Path(manifest['source_zip_path'])
    with ZipFile(source_zip) as archive, gt_path.open('w',encoding='utf-8') as gt:
        for r in records:
            if r['label_zip_entry'] is None:
                failures.append({'sequence_id':r['sequence_id'],'reason':'MISSING_GT'})
                continue
            with TemporaryDirectory(prefix='skyguard-mcos-') as work:
                try:
                    video = Path(work)/'video.mp4'
                    labels = Path(work)/'labels.mat'
                    with archive.open(r['video_zip_entry']) as source, video.open('wb') as target:
                        shutil.copyfileobj(source,target)
                    with archive.open(r['label_zip_entry']) as source, labels.open('wb') as target:
                        shutil.copyfileobj(source,target)
                    bboxes = load_groundtruth(str(labels))
                    if not isinstance(bboxes, (list,tuple)):
                        raise ValueError('Unexpected MCOS annotation type')
                    cap = cv2.VideoCapture(str(video))
                    if not cap.isOpened():
                        raise ValueError('Could not decode video')
                    frame_index, selected = 0, 0
                    try:
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            if frame_index >= len(bboxes):
                                raise ValueError('Video has more frames than MCOS labels')
                            if frame_index % stride == 0:
                                height, width = frame.shape[:2]
                                bbox = _bbox_to_yolo(bboxes[frame_index], width, height)
                                filename = f"{r['sequence_id']}__f{frame_index:07d}"
                                image_out = root/'images'/r['split']/(filename+'.jpg')
                                label_out = root/'labels'/r['split']/(filename+'.txt')
                                image_out.parent.mkdir(parents=True,exist_ok=True)
                                label_out.parent.mkdir(parents=True,exist_ok=True)
                                if not cv2.imwrite(str(image_out), frame):
                                    raise ValueError('Image write failed')
                                if bbox is None:
                                    label_out.write_text('',encoding='ascii')
                                else:
                                    cls_id = CLASSES.index(r['class'])
                                    label_out.write_text(str(cls_id)+' '+' '.join(f'{x:.8f}' for x in bbox)+'\n',encoding='ascii')
                                record = {'sequence_id':r['sequence_id'], 'frame_index':frame_index,
                                          'class':r['class'], 'modality':r['modality'], 'split':r['split'],
                                          'image':str(image_out.relative_to(root)),
                                          'bbox_yolo':bbox, 'width':width, 'height':height,
                                          'gt_track_id':r['sequence_id'] if bbox is not None else None}
                                gt.write(json.dumps(record,ensure_ascii=False)+'\n')
                                stats[f"frames_{r['split']}"] += 1
                                if bbox is None:
                                    stats['background_frames'] += 1
                                else:
                                    stats[f"objects_{r['class']}"] += 1
                                selected += 1
                                if max_frames_per_video and selected >= max_frames_per_video:
                                    break
                            frame_index += 1
                    finally:
                        cap.release()
                    if not max_frames_per_video and frame_index < len(bboxes)-1:
                        raise ValueError(f'MCOS/video mismatch: video={frame_index}, annotations={len(bboxes)}')
                    stats['sequences_exported'] += 1
                except Exception as ex:
                    # Fail non-silently, and mark source as unusable for a valid report.
                    failures.append({'sequence_id':r['sequence_id'], 'reason':f'{type(ex).__name__}: {ex}'})
    data_yaml = ('path: '+root.resolve().as_posix()+'\n'
                 'train: images/train\nval: images/val\ntest: images/test\n'
                 'names:\n' + ''.join(f'  {i}: {name.lower()}\n' for i,name in enumerate(CLASSES)))
    (root/'data.yaml').write_text(data_yaml,encoding='utf-8')
    report = {'source_sha256':manifest['source_sha256'], 'class_mapping':list(CLASSES),
              'statistics':dict(stats), 'errors':failures,
              'status':'PASS_EXPORT' if not failures and stats['sequences_exported'] else 'FAIL_EXPORT',
              'limits':{'stride':stride,'max_sequences':max_sequences,
                        'max_frames_per_video':max_frames_per_video},
              'warning':'Halmstad single-object clips do not validate multi-object IDF1/HOTA'}
    json_dump_atomic(report, root/'export_report.json')
    return report


def train_4class(data_yaml: str | Path, *, epochs: int = 35, imgsz: int = 640,
                 model: str = 'yolo11n.pt', device: str = 'cpu',
                 output_dir: str = 'backend/reports/phase43_training') -> dict:
    if epochs <= 0 or imgsz < 32:
        raise ValueError('Invalid training settings')
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install: pip install ultralytics') from exc
    src = Path(data_yaml)
    if not src.is_file():
        raise FileNotFoundError(src)
    result = YOLO(model).train(data=str(src.resolve()), epochs=epochs,
                               imgsz=imgsz, device=device,
                               project=str(Path(output_dir).resolve()), name='four_class',
                               exist_ok=False, seed=43, deterministic=True)
    return {'status':'TRAINING_COMPLETED_NOT_VALIDATED',
            'run_dir':str(result.save_dir) if hasattr(result,'save_dir') else output_dir}
