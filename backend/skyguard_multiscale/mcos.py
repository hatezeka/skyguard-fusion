"""Read-only video/MCOS evidence audit. NEVER silently align an annotation offset.

Count equality alone does not establish spatial or temporal correspondence.
An approved alignment must be reviewed against visual ground truth before training.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json, shutil


def inspect_record(archive, row):
    import cv2
    from mcos_decoder import load_groundtruth
    decoded = 0
    annotation_count = None
    cap = None
    try:
        if not row.get('label_zip_entry'):
            raise ValueError('MISSING_GROUND_TRUTH')
        with TemporaryDirectory(prefix='skyguard-mcos-audit-') as directory:
            video = Path(directory) / 'video.mp4'
            labels = Path(directory) / 'labels.mat'
            for zip_key, destination in (('video_zip_entry', video), ('label_zip_entry', labels)):
                with archive.open(row[zip_key]) as src, destination.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            boxes = load_groundtruth(str(labels))
            if not isinstance(boxes, (list, tuple)):
                raise ValueError('INVALID_ANNOTATION_FORMAT')
            annotation_count = len(boxes)
            try:
                cap = cv2.VideoCapture(str(video))
                if not cap.isOpened():
                    raise ValueError('VIDEO_OPEN_FAILED')
                while True:
                    ok, _ = cap.read()
                    if not ok: break
                    decoded += 1
            finally:
                if cap is not None:
                    cap.release()  # MUST happen before temp directory exits on Windows
                    cap = None
        if decoded == 0:
            raise ValueError('EMPTY_VIDEO')
        diff = decoded - annotation_count
        if diff == 0:
            state = 'COUNT_MATCH_ALIGNMENT_UNVERIFIED'
        elif abs(diff) == 1:
            state = 'QUARANTINE_OFF_BY_ONE'
        else:
            state = 'QUARANTINE_MAJOR_MISMATCH'
        return dict(sequence_id=row['sequence_id'], split=row['split'],
                    class_name=row['class'], modality=row['modality'],
                    decoded_frames=decoded, annotation_frames=annotation_count,
                    difference=diff, state=state)
    except Exception as exc:
        return dict(sequence_id=row['sequence_id'], split=row['split'],
                    class_name=row['class'], modality=row['modality'],
                    decoded_frames=decoded, annotation_frames=annotation_count,
                    difference=None, state='QUARANTINE_ERROR',
                    error=f'{type(exc).__name__}: {exc}')


def audit(manifest_path: str, report_path: str):
    path = Path(report_path)
    if path.exists():
        raise FileExistsError(path)
    data = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    manifest = data.get('halmstad', data)
    rows = manifest['records']
    if len({r['sequence_id'] for r in rows}) != len(rows):
        raise ValueError('DUPLICATE_SEQUENCE_IDS')
    outcomes = []
    with ZipFile(manifest['source_zip_path']) as archive:
        for i, row in enumerate(rows, 1):
            outcomes.append(inspect_record(archive, row))
            if i % 25 == 0 or i == len(rows):
                print(f'AUDITED {i}/{len(rows)}', flush=True)
    assert len(outcomes) == len(rows)
    state_counts = dict(Counter(r['state'] for r in outcomes))
    group_counts = [dict(split=s, class_name=c, modality=m, state=st, count=n)
                    for (s,c,m,st),n in sorted(Counter(
                        (r['split'],r['class_name'],r['modality'],r['state'])
                        for r in outcomes).items())]
    result = dict(status='AUDITED_NOT_TRAIN_READY', source_sha256=manifest['source_sha256'],
                  total=len(rows), state_counts=state_counts, groups=group_counts,
                  outcomes=outcomes,
                  scientific_limits=['Count match is not proof of aligned labels',
                                     'No off-by-one compensation has been applied',
                                     'No missing frame labels have been synthesized',
                                     'No visual bbox alignment or leakage verification completed'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def propose_clean_manifest(manifest_path, audit_path, output_path):
    """Only count-matching candidates. NOT an automatically certified training set."""
    result = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    data = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    manifest = data.get('halmstad', data)
    if result['source_sha256'] != manifest['source_sha256']:
        raise ValueError('SOURCE_HASH_MISMATCH')
    if result['total'] != len(manifest['records']):
        raise ValueError('AUDIT_INCOMPLETE')
    allowed = {row['sequence_id'] for row in result['outcomes']
               if row['state'] == 'COUNT_MATCH_ALIGNMENT_UNVERIFIED'}
    clean = dict(manifest, records=[r for r in manifest['records']
                                    if r['sequence_id'] in allowed])
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean, indent=2), encoding='utf-8')
    return {'candidate_sequences':len(clean['records']),
            'status':'CANDIDATE_ONLY_NOT_TRAIN_READY',
            'class_split_coverage': [dict(split=s, class_name=c,
                                           count=sum(1 for r in clean['records']
                                                     if r['split']==s and r['class']==c))
                                     for s in ('train','val','test')
                                     for c in ('DRONE','BIRD','AIRPLANE','HELICOPTER')]}


def render_alignment_evidence(manifest_path, sequence_id, output_dir, *, offsets=(-1,0,1), sample_count=5):
    """Render candidate offset overlays for HUMAN inspection, never auto-approve.

    The offset is annotation_index = decoded_frame_index + offset.
    Example with N+1 labels: offset=0 and offset=1 are both possible.
    Comparing the overlays against actual object location is necessary;
    a contact sheet is evidence for sampled frames, not proof for the full clip.
    """
    import cv2
    from mcos_decoder import load_groundtruth
    source=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    manifest=source.get('halmstad',source)
    entries=[r for r in manifest['records'] if r['sequence_id']==sequence_id]
    if len(entries)!=1:
        raise ValueError('SEQUENCE_ID_MISSING_OR_DUPLICATE')
    row=entries[0]
    target=Path(output_dir)
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True,exist_ok=False)
    emitted=[]
    with ZipFile(manifest['source_zip_path']) as archive, TemporaryDirectory(prefix='skyguard-mcos-visual-') as directory:
        video=Path(directory)/'video.mp4'; label=Path(directory)/'labels.mat'
        for key,path in (('video_zip_entry',video),('label_zip_entry',label)):
            with archive.open(row[key]) as src,path.open('wb') as dst:
                shutil.copyfileobj(src,dst)
        boxes=load_groundtruth(str(label))
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError('VIDEO_OPEN_FAILED')
        try:
            decoded=[]
            while True:
                ok,frame=cap.read()
                if not ok:break
                decoded.append(frame)
        finally:
            cap.release()
        if not decoded:
            raise ValueError('NO_DECODED_VIDEO_FRAMES')
        indices=sorted(set(round(i*(len(decoded)-1)/(sample_count-1))
                           for i in range(sample_count))) if sample_count>1 else [0]
        for index in indices:
            for offset in offsets:
                annotated_index=index+offset
                image=decoded[index].copy()
                valid=0<=annotated_index<len(boxes)
                box=boxes[annotated_index] if valid else None
                if valid and box is not None:
                    try:
                        x,y,w,h=(float(v) for v in box)
                        if not all(__import__('math').isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0:
                            raise ValueError('INVALID_BBOX_GEOMETRY')
                        cv2.rectangle(image,(round(x),round(y)),(round(x+w),round(y+h)),(0,255,255),2)
                    except (ValueError,TypeError) as exc:
                        raise ValueError(f'INVALID_MCOS_BBOX: frame={index} ann={annotated_index}') from exc
                label_text=f'frame {index}, MCOS {annotated_index}, offset {offset}, in_range={valid}'
                cv2.putText(image,label_text,(5,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                file=target/f'frame_{index:06d}__offset_{offset:+d}.jpg'
                if not cv2.imwrite(str(file),image):
                    raise ValueError('EVIDENCE_IMAGE_WRITE_FAILED')
                emitted.append(dict(frame_index=index, annotation_index=annotated_index,
                                    offset=offset, in_range=valid, file=file.name))
    summary=dict(status='SAMPLE_VISUAL_EVIDENCE_REVIEW_REQUIRED',
                 sequence_id=sequence_id,source_sha256=manifest['source_sha256'],
                 decoded_frames=len(decoded),annotation_frames=len(boxes),
                 candidate_offsets=list(offsets),sampled_pairs=emitted,
                 caution='Manual visual review required; no offset is automatically certified.')
    (target/'evidence.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
