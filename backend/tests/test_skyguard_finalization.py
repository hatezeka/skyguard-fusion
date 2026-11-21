"""Scientific release guard tests, executed with the full backend suite."""
import json
import pytest
from app.skyguard_finalization.scientific import iou, score_ground_truth, validate_samples


def frame(i, cls='BIRD', track=1, bbox=None):
    return {'sequence_id':'s','frame_index':i,'objects':[{'class':cls,'track_id':track,'bbox':bbox or [0,0,10,10]}]}


def test_iou_exact():
    assert iou([0,0,10,10],[0,0,10,10])==1
    assert iou([0,0,10,10],[20,20,30,30])==0


def test_rejects_nan():
    with pytest.raises(ValueError):iou([0,0,10,float('nan')],[0,0,10,10])


def test_rejects_unknown_track():
    with pytest.raises(ValueError):validate_samples([frame(0,'UNKNOWN',1)])


def test_unknown_without_track_valid():
    validate_samples([frame(0,'UNKNOWN',None)])


def test_rejects_duplicate_sequence_frame():
    with pytest.raises(ValueError):validate_samples([frame(0),frame(0)])


def test_rejects_duplicate_ids_in_one_frame():
    x=frame(0); x['objects'].append(dict(x['objects'][0]))
    with pytest.raises(ValueError):validate_samples([x])


def test_global_identity_permutation_is_correct():
    gt=[frame(i,track=1) for i in range(5)]
    pred=[frame(i,track=99) for i in range(5)]
    r=score_ground_truth(gt,pred)
    assert r['idf1']==pytest.approx(1.)
    assert r['precision']==pytest.approx(1.)


def test_id_switch_penalized():
    gt=[frame(i,track=1) for i in range(4)]
    pred=[frame(i,track=(1 if i<2 else 2)) for i in range(4)]
    assert score_ground_truth(gt,pred)['idf1']<1.


def test_no_fake_idf1_without_ground_truth_ids():
    gt=[frame(i,track=None) for i in range(3)]
    pred=[frame(i,track=7) for i in range(3)]
    assert score_ground_truth(gt,pred)['idf1'] is None


def test_class_confusion_not_scored_as_match():
    r=score_ground_truth([frame(0,cls='BIRD')],[frame(0,cls='DRONE')])
    assert (r['tp'],r['fp'],r['fn'])==(0,1,1)
    assert r['class_confusion']['BIRD->DRONE']==1


def test_prediction_outside_gt_rejected():
    with pytest.raises(ValueError):score_ground_truth([frame(0)],[frame(1)])


def test_no_hota_claim():
    r=score_ground_truth([frame(0)],[frame(0)])
    assert r['hota'] is None
    assert 'NOT_IMPLEMENTED' in r['hota_status']


def test_tracker_guard_installed():
    from pathlib import Path
    backend=Path(__file__).resolve().parents[1]
    s=(backend/'app/detection/live_session.py').read_text(encoding='utf-8-sig')
    t=(backend/'app/detection/tracker.py').read_text(encoding='utf-8-sig')
    assert 'tracked_detections = [' in s
    assert 'detections=tracked_detections,' in s
    assert 'track.detection.object_class != detection.object_class' in t


def test_production_tracker_never_reuses_id_across_classes():
    from app.detection.tracker import ObjectTracker
    from app.detection.engine import Detection, BoundingBox
    tracker=ObjectTracker(min_confirmed_hits=1)
    box=BoundingBox(0,0,10,10)
    bird=Detection('bird',.9,box)
    drone=Detection('drone',.9,box)
    first=tracker.update(0,[bird])[0]
    second=next(o for o in tracker.update(1,[drone]) if o.detection.object_class=='drone')
    assert first.track_id!=second.track_id


def test_three_class_manifest_requires_rights(tmp_path):
    from app.skyguard_finalization.training_3class import audit_dataset
    with pytest.raises(ValueError):
        audit_dataset(tmp_path,{'items':[{'split':'train','image':'a.jpg','label':'a.txt','source_group':'a',
                                          'license':'unverified','annotation_reviewed':True}]})


def test_three_class_manifest_rejects_leakage(tmp_path):
    from app.skyguard_finalization.training_3class import audit_dataset
    (tmp_path/'a.jpg').write_bytes(b'fixture')
    (tmp_path/'a.txt').write_text('2 0.5 0.5 0.2 0.2\n')
    items=[{'split':s,'image':'a.jpg','label':'a.txt','source_group':'same source',
            'license':'approved','annotation_reviewed':True} for s in ('train','val','test')]
    with pytest.raises(ValueError,match='leakage'):
        audit_dataset(tmp_path,{'items':items})


def test_three_class_dataset_source_disjoint_passes(tmp_path):
    from app.skyguard_finalization.training_3class import audit_dataset
    items=[]
    for idx,(split,cls) in enumerate([('train',2),('val',1),('test',2)]):
        image=f'frame{idx}.jpg'; ann=f'frame{idx}.txt'
        (tmp_path/image).write_bytes(('fixture'+str(idx)).encode())
        (tmp_path/ann).write_text(f'{cls} 0.5 0.5 0.2 0.2\n')
        items.append({'split':split,'image':image,'label':ann,'source_group':f'independent_source_{idx}',
                      'license':'approved by fixture','annotation_reviewed':True})
    result=audit_dataset(tmp_path,{'items':items})
    assert result['source_groups']==3
    assert result['counts']['test:other_airborne']==1
