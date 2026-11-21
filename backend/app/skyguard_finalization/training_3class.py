"""Strict source-disjoint three-class YOLO dataset gate and opt-in training.

No pseudo-labels from filenames, no assumption 'negative' means verified OTHER,
no training without verified annotation provenance and explicit rights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from collections import Counter,defaultdict
from pathlib import Path

REQUIRED=['drone','bird','other_airborne']
SPLITS={'train','val','test'}
EXTS={'.jpg','.jpeg','.png','.webp','.bmp'}


def hash_file(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def valid_labels(path:Path):
    if not path.is_file():raise ValueError(f'Missing reviewed label file: {path.name}')
    counts=Counter()
    for line_number,line in enumerate(path.read_text(encoding='utf8').splitlines(),1):
        if not line.strip():continue
        parts=line.split()
        if len(parts)!=5:raise ValueError(f'Invalid YOLO box at {path.name}:{line_number}')
        try: cls=int(parts[0]); vals=[float(s) for s in parts[1:]]
        except ValueError as exc:raise ValueError(f'Invalid YOLO numeric value at {path.name}:{line_number}') from exc
        if str(cls)!=parts[0] or cls not in (0,1,2):raise ValueError(f'Unexpected class at {path.name}:{line_number}')
        x,y,w,h=vals
        if not all(math.isfinite(v) for v in vals) or not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
            raise ValueError(f'Invalid normalized box at {path.name}:{line_number}')
        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2>1+1e-6 or y+h/2>1+1e-6:
            raise ValueError(f'Box out of image bounds at {path.name}:{line_number}')
        counts[cls]+=1
    return counts


def audit_dataset(root:Path,manifest:dict):
    """Manifest is independently reviewed evidence, not generated from folder names.

    items: [{split, image, label, source_group, license, annotation_reviewed}]
    """
    entries=manifest.get('items')
    if not isinstance(entries,list) or not entries:raise ValueError('Non-empty manifest items required')
    if len(entries)>1_000_000:raise ValueError('Unbounded dataset')
    groups=defaultdict(set); hashes=defaultdict(set); counts=Counter(); images=set(); labels=set()
    for item in entries:
        if not isinstance(item,dict):raise ValueError('Manifest item must be mapping')
        split=item.get('split'); group=item.get('source_group')
        if split not in SPLITS:raise ValueError('Unknown split')
        if not isinstance(group,str) or not group.strip():raise ValueError('Source group required')
        if item.get('annotation_reviewed') is not True:raise ValueError('Human review of every annotation required')
        if not isinstance(item.get('license'),str) or item['license'].lower() in ('','unknown','unverified'):
            raise ValueError('Documented source license/provenance required')
        groups[group].add(split)
        for key,used in [('image',images),('label',labels)]:
            rel=item.get(key)
            if not isinstance(rel,str) or not rel or Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError('Unsafe or missing dataset relative path')
            path=(root/rel).resolve()
            if root.resolve() not in path.parents:raise ValueError('Dataset path outside source root')
            identity=(split,str(path))
            if identity in used:raise ValueError('Duplicate image or annotation in manifest')
            used.add(identity)
        image=(root/item['image']).resolve(); label=(root/item['label']).resolve()
        if image.suffix.lower() not in EXTS or not image.is_file():raise ValueError('Image file missing/invalid')
        if label.suffix.lower()!='.txt':raise ValueError('Annotation must be YOLO txt')
        hashes[hash_file(image)].add(split)
        counts.update({f'{split}:images':1})
        for cls,n in valid_labels(label).items():counts[f'{split}:{REQUIRED[cls]}']+=n
    leakage={k:sorted(v) for k,v in groups.items() if len(v)>1}
    duplicate_images={k:sorted(v) for k,v in hashes.items() if len(v)>1}
    if leakage:raise ValueError(f'Source-group train/val/test leakage (first 3): {list(leakage)[:3]}')
    if duplicate_images:raise ValueError('Identical image content leaks across splits')
    for split in SPLITS:
        if not counts[f'{split}:images']:raise ValueError(f'Empty split {split}')
    if not counts['train:other_airborne'] or not counts['test:other_airborne']:
        raise ValueError('Reviewed OTHER bounding boxes required in train AND held-out test')
    return {'status':'DATASET_STRUCTURE_AUDIT_PASS_NOT_MODEL_ACCURACY',
            'counts':dict(counts),'source_groups':len(groups),'images':len(images),
            'gt_rights':'MANIFEST_ATTESTED_NOT_INDEPENDENTLY_VERIFIED'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-yaml',required=True,type=Path)
    p.add_argument('--source-manifest',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--confirm-rights',action='store_true')
    p.add_argument('--train',action='store_true')
    p.add_argument('--base-model',type=Path)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    report={'status':'BLOCKED'}
    rc=1
    try:
        if not a.confirm_rights:raise PermissionError('Dataset permission confirmation is required')
        import yaml
        config=yaml.safe_load(a.data_yaml.read_text(encoding='utf8'))
        names=config.get('names')
        if isinstance(names,dict):names=[names[i] if i in names else names[str(i)] for i in range(3)]
        if config.get('nc')!=3 or [str(x).lower() for x in names]!=REQUIRED:
            raise ValueError('data.yaml must have nc=3 and ordered names [drone,bird,other_airborne]')
        base=a.data_yaml.parent.resolve()
        source_root=Path(config.get('path',base))
        if not source_root.is_absolute():source_root=(base/source_root)
        source_root=source_root.resolve()
        manifest=json.loads(a.source_manifest.read_text(encoding='utf8'))
        report=audit_dataset(source_root,manifest)
        report['dataset_yaml_sha256']=hash_file(a.data_yaml)
        report['manifest_sha256']=hash_file(a.source_manifest)
        if a.train:
            if not a.base_model or not a.base_model.is_file():raise ValueError('Verified base model required')
            if a.epochs<1 or a.epochs>300:raise ValueError('epochs outside 1..300')
            from ultralytics import YOLO
            trained=YOLO(str(a.base_model)).train(data=str(a.data_yaml),epochs=a.epochs,
                         device=a.device,imgsz=640,project=str(a.report.parent/'training_runs'),
                         name='tri_class',exist_ok=False)
            report['status']='TRAINING_EXECUTED_EVALUATION_REQUIRED'
            report['training_return_type']=type(trained).__name__
            report['base_checkpoint_sha256']=hash_file(a.base_model)
        rc=0
    except Exception as exc:
        report['status']='BLOCKED'
        report['error_kind']=type(exc).__name__
        report['error']=str(exc)[:600]
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'report':str(a.report),'error':report.get('error')}))
    raise SystemExit(rc)

if __name__=='__main__':main()
