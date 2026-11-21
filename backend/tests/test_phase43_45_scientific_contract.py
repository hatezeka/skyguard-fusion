"""Phase 43-45 independent contract tests: no false IDs or geolocation."""
import math
import pytest
from app.phase43_45.perception import (Box, Candidate, DecisionGate,
    TargetOnlyTracker, model_capability, iou)
from app.phase43_45.maps import dual_scene, pixel_bearing, point_from_verified_range
from app.phase43_45.evaluate import evaluate_groundtruth
from app.phase43_45.benchmark import percentile
from app.phase43_45.acceptance import build_matrix,field_template


def det(cls='DRONE', x=20., y=20., w=10., h=10., confidence=.9):
    return Candidate(cls, confidence, Box(x,y,w,h))


def test_two_class_model_cannot_admit_target_to_tracker():
    gate=DecisionGate(['drone','bird'])
    decision=gate.decide(det())
    assert decision['class']=='UNKNOWN'
    assert decision['target_eligible'] is False
    assert 'TWO_CLASS' in decision['reject_reason']


def test_three_class_other_airborne_checkpoint_supported():
    assert model_capability(['drone','bird','other_airborne'])['open_set_candidate']


def test_four_class_recognizes_explicit_negatives():
    gate=DecisionGate(['drone','bird','airplane','helicopter'])
    assert gate.capability['open_set_candidate']
    for label in ['AIRPLANE','HELICOPTER','OTHER','UNKNOWN']:
        d=gate.decide(det(label))
        assert d['class']=='UNKNOWN' and not d['target_eligible']
    assert gate.decide(det('DRONE'))['class']=='DRONE'
    assert gate.decide(det('BIRD',confidence=.5))['class']=='UNKNOWN'


def test_reject_invalid_confidence_and_box():
    gate=DecisionGate(['DRONE','BIRD','AIRPLANE','HELICOPTER'])
    with pytest.raises(ValueError):gate.decide(det(confidence=math.nan))
    with pytest.raises(ValueError):gate.decide(det(w=0))
    with pytest.raises(ValueError):DecisionGate(['DRONE'],{'DRONE':2})


def test_unknown_never_receives_id_nor_trajectory():
    gate=DecisionGate(['DRONE','BIRD','AIRPLANE','HELICOPTER'])
    tracker=TargetOnlyTracker(min_hits=2)
    for frame in range(1,5):
        raw=[det('AIRPLANE',20+frame),det('BIRD',100+frame)]
        filtered=[gate.decide(x) for x in raw]
        result=tracker.update(frame,filtered)
        assert all(t['class']=='BIRD' for t in result['confirmed'])
        assert all(t['track_id'] > 0 for t in result['confirmed'])
    assert len(tracker.tracks)==1


def test_tracking_continuity_and_strict_frame_order():
    gate=DecisionGate(['DRONE','BIRD','AIRPLANE','HELICOPTER'])
    tracker=TargetOnlyTracker(min_hits=2)
    assert tracker.update(1,[gate.decide(det())])['confirmed']==[]
    result=tracker.update(2,[gate.decide(det(x=21))])
    assert len(result['confirmed'])==1
    tid=result['confirmed'][0]['track_id']
    assert tracker.update(3,[gate.decide(det(x=22))])['confirmed'][0]['track_id']==tid
    with pytest.raises(ValueError):tracker.update(3,[])


def test_tracker_unknown_expire_and_no_cross_class_association():
    gate=DecisionGate(['DRONE','BIRD','AIRPLANE','HELICOPTER'])
    tracker=TargetOnlyTracker(min_hits=1,max_gap=0)
    target=gate.decide(det('DRONE'))
    assert tracker.update(1,[target])['confirmed'][0]['track_id']==1
    bird=gate.decide(det('BIRD'))
    assert tracker.update(2,[bird])['confirmed'][0]['track_id']==2
    tracker.update(3, [])
    assert 1 not in tracker.tracks


def test_ambiguous_association_suppresses_ids():
    gate=DecisionGate(['DRONE','BIRD','AIRPLANE','HELICOPTER'])
    tr=TargetOnlyTracker(min_hits=1,ambiguity_delta=.5)
    tr.update(0,[gate.decide(det(x=10)),gate.decide(det(x=16))])
    result=tr.update(1,[gate.decide(det(x=13))])
    assert result['confirmed']==[] and result['ambiguous_count']==1


def test_iou_identity():
    assert iou(Box(0,0,10,10),Box(0,0,10,10))==pytest.approx(1)
    assert iou(Box(0,0,10,10),Box(200,200,10,10))==0


def test_no_synthetic_wgs84_or_range():
    track={'track_id':7,'class':'DRONE','bbox':[40,20,10,10]}
    scene=dual_scene('s',5,50.,[track],width=100,hfov_deg=90,
                     camera={'azimuth_deg':30,'latitude_deg':42.5,'longitude_deg':21.1,
                             'georeference_verified':True})
    assert scene['tactical_map']['objects'][0]['track_id']==7
    assert scene['real_map']['objects'][0]['track_id']==7
    assert scene['real_map']['objects'][0]['wgs84'] is None
    assert scene['tactical_map']['objects'][0]['east_m'] is None


def test_known_reference_works_but_requires_verified_provenance():
    camera={'azimuth_deg':0,'latitude_deg':42.,'longitude_deg':21.,'georeference_verified':True}
    with pytest.raises(ValueError):point_from_verified_range(camera,0.,{'source':'YOLO','horizontal_range_m':100,'uncertainty_m':5})
    scene=dual_scene('s',1,100.,[{'track_id':1,'class':'BIRD','bbox':[45,20,10,10]}],100,
               hfov_deg=90,camera=camera,
               measured_ranges={1:{'source':'RANGEFINDER','horizontal_range_m':100,
                                    'uncertainty_m':2}})
    obj=scene['real_map']['objects'][0]
    assert obj['wgs84'] is not None
    assert obj['uncertainty_m']==2
    assert scene['tactical_map']['objects'][0]['north_m']==pytest.approx(100)


def test_unknown_refused_as_map_track():
    with pytest.raises(ValueError):dual_scene('s',1,0,[{'track_id':1,'class':'UNKNOWN','bbox':[1,1,1,1]}],100)


def test_project_bearing_center():
    assert pixel_bearing([45,20,10,10],100,90,15)==pytest.approx(15)
    with pytest.raises(ValueError):pixel_bearing([100,20,10,10],100,90,15)


def test_evaluate_detects_negative_false_drone():
    gt=[{'sequence_id':'plane','frame_index':0,'split':'test','objects':[
        {'class':'AIRPLANE','bbox':[0,0,10,10]}]}]
    pred=[{'sequence_id':'plane','frame_index':0,'split':'test','objects':[
        {'class':'DRONE','bbox':[0,0,10,10]}]}]
    result=evaluate_groundtruth(gt,pred)
    assert result['class_confusion']['AIRPLANE->DRONE']==1
    assert result['negative_drone_false_alarm_frame_rate']==1.
    assert result['idf1'] is None
    assert result['HOTA'] is None


def test_evaluate_rejects_train_leakage():
    gt=[{'sequence_id':'x','frame_index':0,'split':'train','objects':[]}]
    pred=[{'sequence_id':'x','frame_index':0,'objects':[]}]
    with pytest.raises(ValueError):evaluate_groundtruth(gt,pred)


def test_evaluate_reports_exact_idf1():
    gt=[{'sequence_id':'x','frame_index':i,'split':'test','objects':[
        {'class':'BIRD','bbox':[0,0,10,10],'track_id':1}]} for i in range(2)]
    pr=[{'sequence_id':'x','frame_index':i,'objects':[
        {'class':'BIRD','bbox':[0,0,10,10],'track_id':10}]} for i in range(2)]
    r=evaluate_groundtruth(gt,pr)
    assert r['idf1']==pytest.approx(1.)
    assert r['id_switches_observed']==0


def test_percentile_real_interpolation():
    assert percentile([1,2,3,4,5],95)==pytest.approx(4.8)
    assert percentile([],95) is None


def test_acceptance_does_not_fabricate_pass(tmp_path):
    r=build_matrix(tmp_path)
    assert r['overall']=='NOT_YET_CERTIFIED'
    assert r['criteria']['usb_field']['status']=='FIELD_PENDING'
    assert r['criteria']['drone_field']['status']=='FIELD_PENDING'
    assert 'PASS' not in {e['status'] for e in r['criteria'].values()}
    assert field_template()['measurements']['range_mae_m'] is None


def test_production_proxy_filters_unknown_while_preserving_tracker_result():
    from app.phase43_45.production_adapter import TargetOnlyTrackerProxy
    from types import SimpleNamespace
    class Spy:
        def reset(self): pass
        def update(self,frame_index,detections): return {'f':frame_index,'d':detections}
    proxy=TargetOnlyTrackerProxy(Spy())
    d=[SimpleNamespace(object_class='drone'),SimpleNamespace(object_class='bird'),
       SimpleNamespace(object_class='unknown'),SimpleNamespace(object_class='airplane'),
       SimpleNamespace(object_class='helicopter'),SimpleNamespace(object_class=None)]
    r=proxy.update(2,d)
    assert r['f']==2 and r['d']==d[:2]
    assert proxy.last_unknown_count==4


def test_calibration_requires_validation_only_and_uses_video_negatives():
    from app.phase43_45.calibration import select_thresholds
    rows=[]
    for i in range(40):
        rows.append({'split':'val','sequence_id':f'drone{i//2}',
            'ground_truth_class':'DRONE','ground_truth_bbox':[0,0,10,10],
            'predictions':[{'class':'DRONE','confidence':.95,'bbox':[0,0,10,10]}]})
        rows.append({'split':'val','sequence_id':f'bird{i//2}',
            'ground_truth_class':'BIRD','ground_truth_bbox':[0,0,10,10],
            'predictions':[{'class':'BIRD','confidence':.95,'bbox':[0,0,10,10]}]})
    for i in range(20):
        rows.append({'split':'val','sequence_id':f'other{i}',
            'ground_truth_class':'HELICOPTER','ground_truth_bbox':[0,0,10,10],
            'predictions':[{'class':'DRONE','confidence':.20,'bbox':[0,0,10,10]}]})
    result=select_thresholds(rows)
    assert result['status']=='VALIDATION_CALIBRATED'
    assert result['criteria']['DRONE']['threshold']>.2
    assert result['criteria']['DRONE']['observed_negative_video_false_alarm_rate']==0
    rows[0]['split']='test'
    with pytest.raises(ValueError):select_thresholds(rows)
