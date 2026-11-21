"""python -m app.phase43_45.cli --help (execute from backend directory)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .inventory import inspect_halmstad_zip, inspect_dut_zip, json_dump_atomic
from .export_yolo import export_from_archive, train_4class
from .evaluate import evaluate_groundtruth, read_jsonl
from .benchmark import benchmark_exported_dataset
from .acceptance import field_template, build_matrix


def cli(argv=None) -> int:
    p=argparse.ArgumentParser(description='SkyGuard FUSION Phase 43-45 evidence toolkit')
    sub=p.add_subparsers(dest='action',required=True)
    inv=sub.add_parser('inventory',help='Inventory uploaded dataset ZIPs with SHA256')
    inv.add_argument('--halmstad',required=True)
    inv.add_argument('--dut')
    inv.add_argument('--out',required=True)
    export=sub.add_parser('export',help='Extract Halmstad MCOS frames to four-class YOLO')
    export.add_argument('--manifest',required=True)
    export.add_argument('--out',required=True)
    export.add_argument('--stride',type=int,default=15)
    export.add_argument('--max-sequences',type=int,default=0)
    export.add_argument('--max-frames-per-video',type=int,default=0)
    train=sub.add_parser('train',help='Optional four-class YOLO training (time-consuming)')
    train.add_argument('--data-yaml',required=True)
    train.add_argument('--epochs',type=int,default=35)
    train.add_argument('--device',default='cpu')
    train.add_argument('--model',default='yolo11n.pt')
    train.add_argument('--out',default='backend/reports/phase43_training')
    bm=sub.add_parser('benchmark',help='Run real YOLO inference on held-out test split')
    bm.add_argument('--dataset',required=True)
    bm.add_argument('--weights',required=True)
    bm.add_argument('--out',required=True)
    bm.add_argument('--max-frames',type=int,default=0)
    bm.add_argument('--calibration',help='Validated model-hash-bound calibration JSON')
    cal=sub.add_parser('calibrate',help='Fit DRONE/BIRD thresholds on VAL sequences only')
    cal.add_argument('--dataset',required=True)
    cal.add_argument('--weights',required=True)
    cal.add_argument('--out',required=True)
    cal.add_argument('--max-frames',type=int,default=0)
    mp=sub.add_parser('yolo-map',help='Run Ultralytics true mAP on held-out test set')
    mp.add_argument('--dataset',required=True)
    mp.add_argument('--weights',required=True)
    mp.add_argument('--out',required=True)
    mp.add_argument('--device',default='cpu')
    ev=sub.add_parser('evaluate-jsonl',help='Score independently supplied GT vs predictions')
    ev.add_argument('--gt',required=True)
    ev.add_argument('--pred',required=True)
    ev.add_argument('--out',required=True)
    ac=sub.add_parser('acceptance',help='Generate honest acceptance matrix')
    ac.add_argument('--reports',required=True)
    fld=sub.add_parser('field-template',help='Generate blank physical test protocol')
    fld.add_argument('--out',required=True)
    usb=sub.add_parser('usb-field',help='PHASE 45 ONLY: real USB video + measured FPS and hashes')
    usb.add_argument('--camera-index',type=int,required=True)
    usb.add_argument('--weights',required=True)
    usb.add_argument('--out',required=True)
    usb.add_argument('--max-frames',type=int,default=300)
    usb.add_argument('--physical-confirm',action='store_true',help='Required only with real USB camera in Phase 45')
    a=p.parse_args(argv)
    if a.action=='inventory':
        obj={'halmstad':inspect_halmstad_zip(a.halmstad)}
        if a.dut:obj['dut']=inspect_dut_zip(a.dut)
        json_dump_atomic(obj,a.out)
        print(json.dumps({'status':'INVENTORIED_NOT_TRAINED','halmstad':obj['halmstad']['counts'],
                          'dut':obj.get('dut',{}).get('status'),'output':a.out},indent=2))
    elif a.action=='export':
        obj=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
        result=export_from_archive(obj['halmstad'] if 'halmstad' in obj else obj,a.out,
              stride=a.stride,max_sequences=a.max_sequences,
              max_frames_per_video=a.max_frames_per_video)
        print(json.dumps(result,indent=2))
        if result['status']!='PASS_EXPORT':return 2
    elif a.action=='train':
        print(json.dumps(train_4class(a.data_yaml,epochs=a.epochs,device=a.device,
                                      model=a.model,output_dir=a.out),indent=2))
    elif a.action=='benchmark':
        calibration=(json.loads(Path(a.calibration).read_text(encoding='utf-8'))
                     if a.calibration else None)
        result=benchmark_exported_dataset(a.dataset,a.weights,a.out,max_frames=a.max_frames,
                                          calibration=calibration)
        print(json.dumps({'status':result['status'], 'performance':result['performance'],
                          'metrics':result['metrics'], 'report_dir':result['report_dir']},indent=2))
    elif a.action=='calibrate':
        from .calibration import calibrate_exported_dataset
        r=calibrate_exported_dataset(a.dataset,a.weights,a.out,max_frames=a.max_frames)
        print(json.dumps(r,indent=2))
        if r['status'] != 'VALIDATION_CALIBRATED':return 2
    elif a.action=='yolo-map':
        from .yolo_map import evaluate_yolo_map
        print(json.dumps(evaluate_yolo_map(a.dataset,a.weights,a.out,device=a.device),indent=2))
    elif a.action=='evaluate-jsonl':
        r=evaluate_groundtruth(read_jsonl(a.gt),read_jsonl(a.pred))
        json_dump_atomic(r,a.out)
        print(json.dumps(r,indent=2))
    elif a.action=='acceptance':
        r=build_matrix(a.reports)
        print(json.dumps({'overall':r['overall'],
                          'criteria':{k:v['status'] for k,v in r['criteria'].items()}},indent=2))
    elif a.action=='field-template':
        json_dump_atomic(field_template(),a.out)
        print('Field trial template created; status FIELD_PENDING:',a.out)
    elif a.action=='usb-field':
        from .field import usb_camera_trial
        print(json.dumps(usb_camera_trial(camera_index=a.camera_index,
             weights=a.weights, destination=a.out, max_frames=a.max_frames,
             physical_confirmed=a.physical_confirm),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(cli())
