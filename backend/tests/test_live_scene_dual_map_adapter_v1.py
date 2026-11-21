"""42.6D independent contract tests: no physical range/GPS assertions."""
from copy import deepcopy
import pytest
from app.localization.live_scene_dual_map_adapter_v1 import (
    TrustedTrackEvidence, attach_dual_map_state,
)
from app.localization.dual_map_projection_v1 import (
    CameraIntrinsics, SurveyedCameraPose, SiteWGS84Reference, TrackFrameObservation,
)
from app.localization.range_evidence_guard_v1 import RangeAssessment


def scene(*, enriched=False):
    item = {'detection_index': 0, 'object_class': 'drone', 'confidence': .9,
            'azimuth_degrees': 90.0, 'target_location': None, 'target_range_m': None}
    s = {'scene_type': 'live_camera_bearing_map', 'camera_id': 'cam1',
         'session_id': 'session-A', 'frame_index': 7, 'observations': [item],
         'geojson': {'type': 'FeatureCollection', 'features': [
             {'type': 'Feature', 'geometry': {'type': 'LineString',
             'coordinates': [[21.1,42.6],[21.102,42.6]]},
             'properties': {'kind': 'bearing_guide', 'display_only': True,
                            'is_target_position': False}}]},
         'target_gps_available': False, 'target_range_available': False,
         'camera_location_survey_verified': False}
    if enriched:
        s['frame_id'] = 'frame-A-7'
        s['timestamp_ms'] = 7000
        s['observations'][0]['track_id'] = 'T-01'
    return s


def evidence(*, frame='frame-A-7', track='T-01', session_id='session-A',
             timestamp=7000, class_name='drone', confidence=.9,
             site=True, depth=True, pose_verified=True):
    obs = TrackFrameObservation(session_id=session_id,frame_id=frame,
          frame_index=7,timestamp_ms=timestamp,track_id=track,
          object_class=class_name,confidence=confidence,
          center_x_px=100,center_y_px=50)
    intr = CameraIntrinsics(fx_px=100,fy_px=100,cx_px=100,cy_px=50,
          image_width_px=200,image_height_px=100,
          calibration_source='synthetic_matrix_test', verified=True)
    pose = SurveyedCameraPose(camera_east_m=0,camera_north_m=0,camera_up_m=0,
          camera_to_enu=((1,0,0),(0,0,1),(0,-1,0)),
          survey_source='synthetic_pose_test',orientation_source='synthetic_north_test',
          verified=pose_verified)
    distance = RangeAssessment(status='estimated',
          reason='LAB_SYNTHETIC_DEPTH_ONLY',camera_optical_axis_depth_m=100,
          conservative_lower_m=90,conservative_upper_m=110,
          method='CALIBRATED_MONOCULAR_APPARENT_SIZE_PINHOLE',
          frame_id=frame,target_track_id=track) if depth else None
    site_ref = SiteWGS84Reference(latitude_deg=42.66,longitude_deg=21.16,
          altitude_m=200,facility_id='test-facility',
          survey_source='synthetic_site_test',verified=True) if site else None
    return TrustedTrackEvidence(0,obs,intr,pose,distance,site_ref)


def test_default_no_target_gps_or_range():
    s=scene()
    out=attach_dual_map_state(s)
    d=out['dual_map_v1']
    assert not d['position_estimate_present']
    assert not d['geographic_estimate_present']
    assert d['tactical_map']['objects'][0]['position_enu_m'] is None
    assert d['tactical_map']['objects'][0]['track_id'] is None
    assert d['real_map']['verified_estimate_geojson']['features']==[]
    assert out['target_gps_available'] is False
    assert out['target_range_available'] is False
    assert d['legacy_display_geojson']['features'][0]['properties']['display_only']


def test_existing_payload_unchanged_and_deep_copy():
    original=scene()
    before=deepcopy(original)
    out=attach_dual_map_state(original)
    assert original==before
    for key,value in original.items():
        assert out[key] == value
    out['dual_map_v1']['legacy_display_geojson']['features'].clear()
    assert original['geojson']['features']


def test_normal_unknown_object_is_not_relabelled_drone():
    s=scene()
    s['observations'][0]['object_class']='unknown'
    out=attach_dual_map_state(s)
    assert out['dual_map_v1']['tactical_map']['objects'][0]['object_class']=='unknown'


def test_empty_observation_scene():
    s=scene();s['observations']=[]
    assert attach_dual_map_state(s)['dual_map_v1']['tactical_map']['objects']==[]

@pytest.mark.parametrize('change', [
    {'scene_type':'fake'}, {'camera_id':''}, {'frame_index':-1},
    {'frame_index':True}, {'observations':None}, {'geojson':None},
])
def test_bad_scene_fails_closed(change):
    with pytest.raises(ValueError):
        attach_dual_map_state({**scene(), **change})


def test_duplicate_detection_indices_fail_closed():
    s=scene();s['observations'].append(deepcopy(s['observations'][0]))
    with pytest.raises(ValueError,match='DUPLICATE'):attach_dual_map_state(s)

@pytest.mark.parametrize('change',[
   {'confidence':float('nan')},{'confidence':1.2},{'detection_index':True},
   {'object_class':'helicopter'}
])
def test_invalid_observations_rejected(change):
    s=scene();s['observations'][0].update(change)
    with pytest.raises(ValueError):attach_dual_map_state(s)


def test_evidence_requires_timestamp_frame_and_stable_track():
    with pytest.raises(ValueError, match='STALE_OR_UNBOUND'):
        attach_dual_map_state(scene(),trusted_evidence=(evidence(),))

@pytest.mark.parametrize('modify',[
    {'frame':'old-frame'}, {'track':'other'}, {'session_id':'other-session'},
    {'timestamp':7001}, {'class_name':'bird'}, {'confidence':.1}
])
def test_mismatched_evidence_fails_closed(modify):
    with pytest.raises(ValueError):
        attach_dual_map_state(scene(enriched=True),trusted_evidence=(evidence(**modify),))


def test_accepted_evidence_produces_two_matching_views():
    s=scene(enriched=True)
    out=attach_dual_map_state(s,trusted_evidence=(evidence(),))
    dual=out['dual_map_v1']
    obj=dual['tactical_map']['objects'][0]
    assert obj['track_id']=='T-01'
    assert obj['status']=='geographic_position_estimate'
    assert obj['position_enu_m']==pytest.approx([0,100,0])
    assert obj['facility_distance_m']==pytest.approx(100)
    assert obj['facility_distance_depth_bounds_m']==pytest.approx([90,110])
    feature=dual['real_map']['verified_estimate_geojson']['features'][0]
    assert feature['geometry']['coordinates'][1]>42.66
    assert feature['properties']['track_id']=='T-01'
    assert feature['properties']['field_validated'] is False
    assert dual['physical_validation_completed'] is False
    assert out['target_gps_available'] is True
    assert out['target_range_available'] is True


def test_evidence_without_site_is_relative_only():
    out=attach_dual_map_state(scene(enriched=True),trusted_evidence=(evidence(site=False),))
    assert out['dual_map_v1']['position_estimate_present']
    assert not out['dual_map_v1']['geographic_estimate_present']
    assert out['target_gps_available'] is False
    assert out['target_range_available'] is True


def test_evidence_without_depth_is_not_assigned_a_distance():
    out=attach_dual_map_state(scene(enriched=True),trusted_evidence=(evidence(depth=False),))
    obj=out['dual_map_v1']['tactical_map']['objects'][0]
    assert obj['status']=='bearing_only'
    assert obj['facility_distance_m'] is None
    assert obj['target_geojson'] is None
    assert not out['dual_map_v1']['geographic_estimate_present']


def test_unverified_camera_pose_rejected_by_projection():
    out=attach_dual_map_state(scene(enriched=True),trusted_evidence=(evidence(pose_verified=False),))
    obj=out['dual_map_v1']['tactical_map']['objects'][0]
    assert obj['status']=='unknown'
    assert obj['position_enu_m'] is None
    assert obj['target_geojson'] is None


def test_duplicate_evidence_and_wrong_detection_id_rejected():
    with pytest.raises(ValueError):attach_dual_map_state(scene(enriched=True),trusted_evidence=(evidence(), evidence()))
    bad=evidence()
    from dataclasses import replace
    with pytest.raises(ValueError):attach_dual_map_state(scene(enriched=True),trusted_evidence=(replace(bad,detection_index=3),))


def test_evidence_cannot_be_dictionary_from_untrusted_json():
    with pytest.raises(ValueError):attach_dual_map_state(scene(enriched=True),trusted_evidence=({'verified':True},))