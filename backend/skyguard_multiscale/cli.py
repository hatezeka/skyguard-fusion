"""python -m skyguard_multiscale.cli ..."""
import argparse, json, time, hashlib
from pathlib import Path
from .detection import CoarseToFine, YOLOAdapter, IoUTracker
from .mcos import audit, propose_clean_manifest, render_alignment_evidence
from .evaluate import evaluate_frame, load_yolo_labels, merge_stats


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_media(args):
    import cv2
    source=Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)
    dst=Path(args.out)
    if dst.exists():
        raise FileExistsError(dst)
    if not Path(args.weights).is_file():
        raise FileNotFoundError(args.weights)
    detector=CoarseToFine(YOLOAdapter(args.weights), max_tiles=args.max_tiles,
                          accept_confidence=args.accept_confidence)
    tracker=IoUTracker()
    is_image=source.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')
    if is_image:
        frame=cv2.imread(str(source))
        if frame is None:
            raise ValueError('UNREADABLE_IMAGE')
        cap=None
        fps=None
    else:
        cap=cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError('VIDEO_OPEN_FAILED')
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not (0<fps<1000):
            fps=None
    # Create output only AFTER validating source/model and opening media.
    dst.mkdir(parents=True, exist_ok=False)
    n=0; milliseconds=[]; count={'large':0,'medium':0,'small':0}
    try:
        with (dst/'detections.jsonl').open('w',encoding='utf-8') as stream:
            while True:
                if not is_image:
                    ok,frame=cap.read()
                    if not ok: break
                t0=time.perf_counter()
                raw=detector.detect(frame)
                tracked=tracker.update(n,raw)
                elapsed=(time.perf_counter()-t0)*1000
                milliseconds.append(elapsed)
                for det in tracked:
                    count[det.tier]+=1
                    if det.class_name=='UNKNOWN' and det.track_id is not None:
                        raise AssertionError('UNKNOWN_TRACK_LEAK')
                # Video timestamps: prefer observed timestamps only when valid,
                # otherwise fps-derived estimates, and mark their provenance.
                if is_image:
                    ts, origin=None, 'still_image'
                else:
                    measured=float(cap.get(cv2.CAP_PROP_POS_MSEC))
                    if n>0 and measured>0:
                        ts,origin=measured,'decoder_timestamp_ms'
                    elif fps is not None:
                        ts,origin=1000*n/fps,'fps_derived_ms'
                    else:
                        ts,origin=None,'unavailable'
                row=dict(frame_index=n,timestamp_ms=ts,timestamp_source=origin,
                         width=frame.shape[1],height=frame.shape[0],
                         inference_ms=elapsed,
                         detections=[det.record() for det in tracked])
                stream.write(json.dumps(row)+'\n')
                n+=1
                if is_image or (args.max_frames and n>=args.max_frames):break
    finally:
        if cap is not None:cap.release()
    if n==0:
        raise ValueError('NO_DECODED_FRAMES')
    latencies=sorted(milliseconds)
    p95=latencies[max(0,min(len(latencies)-1, __import__('math').ceil(.95*len(latencies))-1))]
    report=dict(status='MEASURED_SOFTWARE_ONLY',source_sha256=sha256(source),
                checkpoint_sha256=sha256(args.weights),frames=n,
                total_detections_by_tier=count,processing_fps=1000*n/sum(milliseconds),
                mean_inference_ms=sum(milliseconds)/n,p95_inference_ms=p95,
                caveats=['Processing FPS excludes image capture, rendering and transmission',
                         'Inference includes full640, full1280 and overlapping tiles',
                         'No ground truth scored: accuracy, IDF1 and HOTA unverified',
                         'No spatial distance inferred from monocular bounding boxes',
                         'UNKNOWN has no track_id or trajectory',
                         'Source timestamps may be estimates'])
    (dst/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def run_score(args):
    """Score image/prediction pairs that were explicitly annotated with YOLO GT.
    Labels must have the same filename stem as the test image.
    """
    import cv2
    rows=[]
    predictions={}
    with Path(args.predictions).open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['frame_index'] in predictions:
                raise ValueError('DUPLICATE_PREDICTION_FRAME')
            predictions[row['frame_index']]=row
    images=sorted(Path(args.images).glob('*.jpg'))
    if len(images)!=len(predictions):
        raise ValueError('IMAGE_PREDICTION_COUNT_MISMATCH')
    # Separate image scoring requires a manifest that maps each image to frame index,
    # and never assumes an alphabetical ordering equals video frame order.
    mapping=json.loads(Path(args.mapping).read_text(encoding='utf-8'))
    if set(mapping)!=set(p.name for p in images):
        raise ValueError('EXPLICIT_FRAME_MAPPING_REQUIRED')
    for image in images:
        index=int(mapping[image.name])
        if index not in predictions:
            raise ValueError('MISSING_PREDICTION_FOR_MAPPED_FRAME')
        height,width=cv2.imread(str(image)).shape[:2]
        gt_file=Path(args.labels)/(image.stem+'.txt')
        if not gt_file.is_file():
            raise FileNotFoundError(gt_file)
        gt=load_yolo_labels(gt_file,width,height)
        rows.append(evaluate_frame(predictions[index]['detections'],gt))
    report=dict(status='EVALUATED_ON_PROVIDED_GT_NOT_COCO_MAP',frames=len(images),
                by_class_and_size=merge_stats(rows))
    target=Path(args.out)
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description='SkyGuard scientific MCOS audit and multi-scale inference')
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit-mcos'); a.add_argument('--manifest',required=True); a.add_argument('--out',required=True)
    a=sub.add_parser('visual-evidence'); a.add_argument('--manifest',required=True); a.add_argument('--sequence',required=True); a.add_argument('--out',required=True)
    a=sub.add_parser('candidate-manifest'); a.add_argument('--manifest',required=True); a.add_argument('--audit',required=True); a.add_argument('--out',required=True)
    a=sub.add_parser('detect'); a.add_argument('--input',required=True); a.add_argument('--weights',required=True); a.add_argument('--out',required=True)
    a.add_argument('--max-frames',type=int,default=0);a.add_argument('--max-tiles',type=int,default=128)
    a.add_argument('--accept-confidence',type=float,default=.60)
    a=sub.add_parser('score');a.add_argument('--predictions',required=True);a.add_argument('--images',required=True)
    a.add_argument('--labels',required=True);a.add_argument('--mapping',required=True);a.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='audit-mcos':
        r=audit(args.manifest,args.out);print(json.dumps({k:v for k,v in r.items() if k!='outcomes'},indent=2))
    elif args.command=='visual-evidence':print(json.dumps(render_alignment_evidence(args.manifest,args.sequence,args.out),indent=2))
    elif args.command=='candidate-manifest':print(json.dumps(propose_clean_manifest(args.manifest,args.audit,args.out),indent=2))
    elif args.command=='detect':run_media(args)
    elif args.command=='score':run_score(args)

if __name__=='__main__':main()
