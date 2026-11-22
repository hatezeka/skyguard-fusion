"""Reproducible image/video inference and evidence, original inputs never modified."""
import argparse
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from dataclasses import asdict
import hashlib,json,statistics,time,os
import cv2
from .core import UltralyticsPredictor,multiscale

EXT_IMAGES={'.jpg','.jpeg','.png','.bmp','.webp'}
EXT_VIDEOS={'.mp4','.avi','.mov','.mkv'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def draw(frame,dets):
    out=frame.copy()
    for d in dets:
        x1,y1,x2,y2=map(int,d.xyxy)
        review = d.verdict_status == 'CONFLICT_REVIEW'
        color = (0, 190, 255) if review else {'DRONE':(0,230,0),'BIRD':(255,200,0),'UNKNOWN':(0,160,255)}[d.label]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,'REVIEW' if review else f'{d.label} {d.confidence:.2f}',(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
    return out

def evaluate_file(source,folder,predictor,max_frames=0,stride=1,include_tiles=True,video_webm=False,on_progress=None,inference_lock=None):
    if stride<1 or max_frames<0:raise ValueError('Bad sample configuration')
    source=Path(source);folder=Path(folder)
    if folder.exists():raise FileExistsError(folder)
    if not source.is_file():raise FileNotFoundError(source)
    if source.suffix.lower() not in EXT_IMAGES|EXT_VIDEOS:raise ValueError('Unsupported input')
    folder.mkdir(parents=True)
    is_image=source.suffix.lower() in EXT_IMAGES
    cap=None;writer=None;frame_count=0;decoded_count=0;times=[];counts=Counter();failures=[]
    src_fps=None
    prediction_file = "prediction.png" if is_image else "prediction.mp4"
    try:
        if is_image:
            frame=cv2.imread(str(source));
            if frame is None:raise ValueError('Image decoding failed')
            frames=[(0,frame)]
        else:
            cap=cv2.VideoCapture(str(source))
            if not cap.isOpened():raise ValueError('Video decoding failed')
            src_fps=float(cap.get(cv2.CAP_PROP_FPS))
            if not 0<src_fps<1000:raise ValueError('Invalid video FPS: refuse fabricated timing')
            def iterate():
                nonlocal decoded_count
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    index=decoded_count
                    decoded_count+=1
                    if index%stride==0:yield index,frame
            frames=iterate()
        with (folder/'detections.jsonl').open('w',encoding='utf-8') as journal:
            for index,frame in frames:
                if max_frames and frame_count>=max_frames:break
                start=time.perf_counter()
                with inference_lock if inference_lock is not None else nullcontext():
                    dets=multiscale(frame,predictor,predictor.names,include_tiles=include_tiles,mode=os.environ.get('SKYGUARD_INFERENCE_MODE','adaptive'))
                elapsed=(time.perf_counter()-start)*1000
                times.append(elapsed)
                for d in dets:
                    if d.verdict_status != 'CONFLICT_REVIEW':
                        counts[d.label] += 1
                journal.write(json.dumps({'source_frame_index':index,'source_time_seconds':None if is_image else index/src_fps,
                       'inference_ms':elapsed,'detections':[asdict(d) for d in dets]},sort_keys=True)+'\n')
                annotated=draw(frame,dets)
                if is_image:
                    if not cv2.imwrite(str(folder/'prediction.png'),annotated):raise RuntimeError('Image output failed')
                else:
                    if writer is None:
                        h,w=annotated.shape[:2]
                        if video_webm:
                            writer = cv2.VideoWriter(str(folder/'prediction.webm'),cv2.VideoWriter_fourcc(*'VP80'),src_fps/stride,(w,h))
                            if writer.isOpened():
                                prediction_file = 'prediction.webm'
                            else:
                                writer.release()
                                writer = None
                        if writer is None:
                            writer = cv2.VideoWriter(str(folder/prediction_file),cv2.VideoWriter_fourcc(*'mp4v'),src_fps/stride,(w,h))
                        if not writer.isOpened():raise RuntimeError('Video writer failed')
                    writer.write(annotated)
                frame_count+=1
                if on_progress is not None:
                    on_progress(frame_count)
        if frame_count==0:raise RuntimeError('No frames processed')
    except Exception as exc:
        failures.append(repr(exc))
        raise
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()
        report={'source':str(source),'source_sha256':sha(source),'model_labels':predictor.names,
               'v2_has_unknown':getattr(predictor,'is_v2',False),'frames_processed':frame_count,
               'prediction_file':prediction_file,
               'decoded_frames_read':decoded_count if not is_image else 1,'frame_stride':stride,
               'detections_by_class':dict(counts),'latency_ms_p50':statistics.median(times) if times else None,
               'latency_ms_p95':sorted(times)[min(len(times)-1,int(.95*len(times)))] if times else None,
               'evaluation_status':'INFERENCE_ONLY_NOT_ACCURACY',
               'consensus_note':'Merged geometric proposals; votes from one model are correlated, not calibrated probabilities',
               'limitations':['Requires bounding box ground truth for mAP/precision/recall',
               'V1 two-class cannot establish UNKNOWN reliability','Frame latency depends on local hardware',
               'No absolute distance/GPS without calibrated geometry and additional range information']}
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int,default=120)
    p.add_argument('--stride',type=int,default=1);p.add_argument('--no-tiles',action='store_true')
    args=p.parse_args()
    if args.output.exists():p.error('Output exists, refusing to overwrite')
    if not args.weights.is_file():p.error('Weights missing')
    model=UltralyticsPredictor(args.weights)
    report=evaluate_file(args.input,args.output,model,args.max_frames,args.stride,not args.no_tiles)
    report['weights_sha256']=sha(args.weights)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
