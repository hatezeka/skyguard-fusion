import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from skyguard_multiscale.detection import Detection, CoarseToFine, IoUTracker, box_iou, tile_windows, tier_for, suppress
from skyguard_multiscale.evaluate import evaluate_frame, merge_stats, load_yolo_labels


class FakeScaleModel:
    def predict(self,img,imgsz,confidence):
        h,w=img.shape[:2]
        if w==1280 and imgsz==640:
            return [('DRONE',.94,(100,100,400,400))]  # large
        if w==1280 and imgsz==1280:
            return [('BIRD',.91,(500,500,580,580))]  # medium
        if w<=640 and imgsz==640:
            return [('DRONE',.92,(4,4,24,24))]  # small in tiles
        return []


def test_multiscale_runs_in_order_and_uses_global_coordinates():
    frame=np.zeros((960,1280,3),dtype=np.uint8)
    out=CoarseToFine(FakeScaleModel(),tile_size=640).detect(frame)
    assert {d.tier for d in out}=={'large','medium','small'}
    assert {d.stage for d in out}=={'full_640','full_1280','tiles_640'}
    assert all(0<=d.xyxy[0]<d.xyxy[2]<=1280 and 0<=d.xyxy[1]<d.xyxy[3]<=960 for d in out)


def test_unknown_never_gets_id():
    tracker=IoUTracker()
    dets=[Detection('UNKNOWN',.99,(1,1,50,50),'small','tiles_640'),
          Detection('DRONE',.92,(100,100,300,300),'large','full_640')]
    first=tracker.update(0,dets)
    second=tracker.update(1,dets)
    assert first[0].track_id is None and second[0].track_id is None
    assert first[1].track_id==second[1].track_id


def test_unknown_rejection_of_airplane():
    class Fake:
        def predict(self,img,imgsz,confidence):
            if imgsz==640 and img.shape[1]==800:
                return [('AIRPLANE',.99,(100,100,450,450))]
            return []
    got=CoarseToFine(Fake()).detect(np.zeros((600,800,3),dtype=np.uint8))
    assert len(got)==1 and got[0].class_name=='UNKNOWN' and got[0].track_id is None


def test_uncertain_drone_is_unknown():
    class Fake:
        def predict(self,img,imgsz,confidence):
            return [('DRONE',.4,(100,100,450,450))] if imgsz==640 and img.shape[1]==800 else []
    got=CoarseToFine(Fake()).detect(np.zeros((600,800,3),dtype=np.uint8))
    assert len(got)==1 and got[0].class_name=='UNKNOWN'


def test_tiles_cover_edges_and_limit_is_explicit():
    windows=tile_windows(1001,987,640,.25)
    assert max(w[2] for w in windows)==1001
    assert max(w[3] for w in windows)==987
    with pytest.raises(ValueError,match='TILE_COVERAGE'):
        CoarseToFine(FakeScaleModel(),max_tiles=1).detect(np.zeros((960,1280,3),dtype=np.uint8))


def test_iou_and_nms():
    assert box_iou((0,0,10,10),(0,0,10,10))==1.0
    det=Detection('DRONE',.8,(0,0,10,10),'small','tiles_640')
    assert len(suppress([det,det]))==1
    assert len(suppress([det,Detection('BIRD',.9,det.xyxy,'small','tiles_640')]))==2


def test_scoring_per_size_and_unknown_rejection():
    gt=[dict(class_name='DRONE',xyxy=(10,10,110,110),tier='large'),
        dict(class_name='BIRD',xyxy=(150,150,160,160),tier='small'),
        dict(class_name='AIRPLANE',xyxy=(200,200,400,400),tier='large')]
    predictions=[dict(class_name='DRONE',confidence=.95,xyxy=(10,10,110,110),tier='large'),
                 dict(class_name='UNKNOWN',confidence=.97,xyxy=(200,200,400,400),tier='large')]
    out=evaluate_frame(predictions,gt)
    assert out['DRONE/large']['tp']==1
    assert out['BIRD/small']['fn']==1
    agg=merge_stats([out])
    assert agg['DRONE/large']['precision']==1
    assert agg['BIRD/small']['recall']==0


def test_no_unmapped_yolo_labels(tmp_path):
    f=tmp_path/'a.txt';f.write_text('4 0.5 0.5 0.2 0.2\n')
    with pytest.raises(ValueError,match='INVALID_YOLO_CLASS'):
        load_yolo_labels(f,100,100)


def test_nonmonotonic_frames_rejected():
    tracker=IoUTracker()
    tracker.update(0,[])
    with pytest.raises(ValueError,match='NONMONOTONIC'):
        tracker.update(0,[])
