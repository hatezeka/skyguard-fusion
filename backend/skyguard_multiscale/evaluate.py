"""Independent per-size object-detection verification using YOLO GT.

This is a fixed IoU=.50 point metric; it is NOT COCO mAP or IDF1/HOTA.
"""
import json
from pathlib import Path
from collections import defaultdict
from .detection import box_iou, tier_for

NAMES = ('DRONE','BIRD','AIRPLANE','HELICOPTER')


def load_yolo_labels(path, width, height):
    out=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts)!=5:
            raise ValueError('INVALID_YOLO_LABEL')
        cls=int(parts[0]); x,y,w,h=map(float,parts[1:])
        if cls not in range(4) or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
            raise ValueError('INVALID_YOLO_CLASS_OR_BOX')
        box=((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)
        if not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
            raise ValueError('OUT_OF_IMAGE_YOLO_BOX')
        out.append(dict(class_name=NAMES[cls], xyxy=box, tier=tier_for(box,width,height)))
    return out


def evaluate_frame(predictions, groundtruth, iou_threshold=.5):
    """Greedy one-to-one matching. UNKNOWN predictions count as misses on targets.
    Non-target airplane/helicopter GT count as negative examples: accepting them
    as DRONE/BIRD is counted as false positive on the predicted target class.
    """
    matched=set(); stats=defaultdict(lambda:dict(tp=0,fp=0,fn=0))
    for p in sorted(predictions, key=lambda r:r['confidence'],reverse=True):
        cls=p['class_name']
        if cls=='UNKNOWN': continue
        tier=p['tier']
        choices=sorted(((box_iou(p['xyxy'],gt['xyxy']),j)
                        for j,gt in enumerate(groundtruth)
                        if j not in matched and gt['class_name']==cls), reverse=True)
        if choices and choices[0][0]>=iou_threshold:
            matched.add(choices[0][1]); stats[(cls,tier)]['tp']+=1
        else:
            stats[(cls,tier)]['fp']+=1
    for j,gt in enumerate(groundtruth):
        if j not in matched and gt['class_name'] in ('DRONE','BIRD'):
            stats[(gt['class_name'],gt['tier'])]['fn']+=1
    return {f'{cls}/{tier}':v for (cls,tier),v in sorted(stats.items())}


def merge_stats(reports):
    totals=defaultdict(lambda:dict(tp=0,fp=0,fn=0))
    for report in reports:
        for k,v in report.items():
            for key in ('tp','fp','fn'):
                totals[k][key]+=v[key]
    output={}
    for key,val in sorted(totals.items()):
        tp,fp,fn=val['tp'],val['fp'],val['fn']
        output[key]=dict(**val, precision=tp/(tp+fp) if tp+fp else None,
                         recall=tp/(tp+fn) if tp+fn else None,
                         f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None)
    return output
