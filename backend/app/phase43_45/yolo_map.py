"""Official Ultralytics validation for true mAP50/mAP50:95 on held-out split."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import platform
from .inventory import hash_file,json_dump_atomic
from .perception import model_capability


def evaluate_yolo_map(dataset_root,weights,out_file,*,imgsz=640,device='cpu'):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install ultralytics in backend venv') from exc
    root=Path(dataset_root); yaml=root/'data.yaml';weights=Path(weights)
    if not yaml.is_file() or not weights.is_file():
        raise FileNotFoundError('Missing data.yaml or model checkpoint')
    model=YOLO(str(weights.resolve()))
    capability=model_capability(model.names)
    results=model.val(data=str(yaml.resolve()),split='test',imgsz=imgsz,
                      device=device,plots=False,verbose=False)
    box=results.box
    record={'status':'MEASURED_HELD_OUT_TEST', 'utc':datetime.now(timezone.utc).isoformat(),
            'model_sha256':hash_file(weights),
            'data_yaml_sha256':hash_file(yaml),
            'source_kind':'exported Halmstad video-frame samples',
            'hardware':platform.platform(),
            'model_capability':capability,
            'precision_mean':float(box.mp), 'recall_mean':float(box.mr),
            'mAP50':float(box.map50), 'mAP50_95':float(box.map),
            'limitation':'mAP on sampled frames from one-object videos; not live camera or MOT validation'}
    json_dump_atomic(record,out_file)
    return record
