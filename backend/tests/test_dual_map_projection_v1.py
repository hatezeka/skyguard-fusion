"""Synthetic, independently checkable geometry tests; NO camera/GPS ground truth."""
from __future__ import annotations

from dataclasses import replace
from math import hypot, sqrt

import pytest

from app.localization.dual_map_projection_v1 import (
    CameraIntrinsics, SurveyedCameraPose, SiteWGS84Reference,
    TrackFrameObservation, project_dual_maps,
)
from app.localization.range_evidence_guard_v1 import (
    ApparentSizeEvidence, BoundedMeasurement, estimate_camera_range,
)


def measurement(n, lo=None, hi=None, source='calibrated-lab-measurement'):
    return BoundedMeasurement(n, n if lo is None else lo, n if hi is None else hi, source)


def depth(frame='frame-01', track='drone-01'):
    return estimate_camera_range(ApparentSizeEvidence(
        focal_px=measurement(800, 760, 840),
        apparent_target_size_m=measurement(0.4, 0.35, 0.45),
        measured_image_size_px=measurement(40, 38, 42),
        calibrated_camera=True, independently_sized_target=True,
        frame_id=frame, target_track_id=track, dimension_axis='width'))


def obs(**k):
    d=dict(session_id='session-A', frame_id='frame-01', frame_index=21,
           timestamp_ms=78900, track_id='drone-01', object_class='drone',
           confidence=.88, center_x_px=320., center_y_px=240.)
    d.update(k)
    return TrackFrameObservation(**d)


def intr(**k):
    d=dict(fx_px=800., fy_px=800., cx_px=320., cy_px=240.,
           image_width_px=640, image_height_px=480,
           calibration_source='verified-intrinsic-calibration-record', verified=True)
    d.update(k)
    return CameraIntrinsics(**d)


def pose(**k):
    # A camera aimed north, whose image +y points down toward -Up.
    d=dict(camera_east_m=10., camera_north_m=5., camera_up_m=3.,
           camera_to_enu=((1.,0.,0.), (0.,0.,1.), (0.,-1.,0.)),
           survey_source='known-facility-relative-ENU',
           orientation_source='measured-rotation', verified=True)
    d.update(k)
    return SurveyedCameraPose(**d)


def site(**k):
    d=dict(latitude_deg=45., longitude_deg=9., altitude_m=125.,
           facility_id='facility-X', survey_source='survey-reference',verified=True)
    d.update(k)
    return SiteWGS84Reference(**d)


def test_projected_local_position_matches_pinhole_equations():
    r=project_dual_maps(obs(),intr(),pose(),depth())
    assert r['status']=='relative_position_only'
    assert r['tactical_map']['position_enu_m']==pytest.approx([10.,13.,3.])
    assert r['tactical_map']['facility_distance_m']==pytest.approx(sqrt(100+169+9))
    assert r['real_map']['target_geojson'] is None
    assert r['field_validated'] is False


def test_two_maps_share_exact_identity_when_geo_is_evidenced():
    r=project_dual_maps(obs(),intr(),pose(),depth(),site())
    assert r['status']=='geographic_position_estimate'
    assert r['frame_id']=='frame-01' and r['track_id']=='drone-01'
    assert r['tactical_map']['timestamp_ms']==r['real_map']['timestamp_ms']==78900
    assert r['tactical_map']['track_id']==r['real_map']['track_id']==r['track_id']
    c=r['real_map']['target_geojson']['geometry']['coordinates']
    assert len(c)==3
    assert 8.99<c[0]<9.01 and 44.99<c[1]<45.01
    assert r['real_map']['target_geojson']['properties']['field_validated'] is False


def test_missing_range_produces_no_target_point_or_numeric_distance():
    r=project_dual_maps(obs(),intr(),pose(),None,site())
    assert r['status']=='bearing_only'
    assert r['tactical_map']['position_enu_m'] is None
    assert r['tactical_map']['facility_distance_m'] is None
    assert r['real_map']['target_geojson'] is None
    assert r['real_map']['camera_geojson'] is not None
    assert r['tactical_map']['bearing_deg_from_true_north']==pytest.approx(0)


def test_range_frame_mismatch_fail_closed_even_if_numeric():
    r=project_dual_maps(obs(),intr(),pose(),depth('other-frame'),site())
    assert r['status']=='bearing_only'
    assert r['reason']=='STALE_OR_FOREIGN_RANGE_EVIDENCE'
    assert r['real_map']['target_geojson'] is None


def test_range_track_mismatch_fail_closed_even_if_numeric():
    r=project_dual_maps(obs(),intr(),pose(),depth(track='other-track'),site())
    assert r['status']=='bearing_only'


@pytest.mark.parametrize('changes',[
    {'verified':False}, {'fx_px':0}, {'fx_px':float('nan')},
    {'calibration_source':''}, {'image_width_px':0}, {'fx_px':True},
    {'cx_px':-1}, {'image_height_px':False},
])
def test_bad_intrinsics_never_emits_map_coordinates(changes):
    r=project_dual_maps(obs(),intr(**changes),pose(),depth(),site())
    assert r['status']=='unknown'
    assert r['tactical_map'] is None and r['real_map'] is None


@pytest.mark.parametrize('changes',[
    {'verified':False}, {'survey_source':''}, {'orientation_source':''},
    {'camera_east_m':float('inf')},
    {'camera_to_enu':((1,0,0),(0,1,0),(0,0,-1))}, # reflection
    {'camera_to_enu':((2,0,0),(0,1,0),(0,0,1))}, # scale
    {'camera_to_enu':((1,0,0),(0,0,1),(0,0,1))}, # nonorthogonal
])
def test_invalid_pose_fails_closed(changes):
    r=project_dual_maps(obs(),intr(),pose(**changes),depth(),site())
    assert r['status']=='unknown'
    assert r['real_map'] is None


@pytest.mark.parametrize('changes',[
    {'center_x_px':640.}, {'center_y_px':-1.},
    {'timestamp_ms':-1}, {'frame_index':False},
    {'confidence':1.5}, {'track_id':''}, {'center_x_px':float('inf')}
])
def test_invalid_observation_fails_closed(changes):
    r=project_dual_maps(obs(**changes),intr(),pose(),depth(),site())
    assert r['status']=='unknown'


@pytest.mark.parametrize('changes',[
    {'verified':False}, {'latitude_deg':95.}, {'longitude_deg':-181.},
    {'survey_source':''}, {'altitude_m':float('nan')}
])
def test_invalid_site_blocks_geo_but_not_relative_enu(changes):
    r=project_dual_maps(obs(),intr(),pose(),depth(),site(**changes))
    assert r['status']=='relative_position_only'
    assert r['tactical_map']['position_enu_m'] is not None
    assert r['real_map']['target_geojson'] is None


def test_off_axis_image_point_changes_position_and_bearing():
    r=project_dual_maps(obs(center_x_px=400.),intr(),pose(),depth())
    assert r['status']=='relative_position_only'
    assert r['tactical_map']['position_enu_m']==pytest.approx([10.8,13.,3.])
    assert r['tactical_map']['bearing_deg_from_true_north']>0


def test_depth_interval_facility_bounds_include_nominal_position():
    r=project_dual_maps(obs(),intr(),pose(),depth())
    lo,hi=r['tactical_map']['facility_distance_depth_bounds_m']
    assert lo<=r['tactical_map']['facility_distance_m']<=hi
    assert hi>lo
    assert r['uncertainty_scope']=='DEPTH_INTERVAL_ONLY_NOT_FULL_UNCERTAINTY'


def test_geodetic_zero_offset_matches_site_and_altitude():
    p=pose(camera_east_m=0.,camera_north_m=0.,camera_up_m=0.)
    r=project_dual_maps(obs(),intr(),p,None,site())
    camera=r['real_map']['camera_geojson']['geometry']['coordinates']
    assert camera==pytest.approx([9.,45.,125.],abs=1e-8)


def test_one_metre_east_at_equator_has_correct_longitude_order():
    p=pose(camera_east_m=1.,camera_north_m=0.,camera_up_m=0.)
    s=site(latitude_deg=0.,longitude_deg=0.,altitude_m=0.)
    r=project_dual_maps(obs(),intr(),p,None,s)
    camera=r['real_map']['camera_geojson']['geometry']['coordinates']
    assert camera[0]==pytest.approx(1./111319.49,rel=.0001)
    assert camera[1]==pytest.approx(0.,abs=1e-8)


def test_site_at_dateline_no_invalid_longitude():
    r=project_dual_maps(obs(),intr(),pose(),depth(),site(longitude_deg=179.99999))
    assert r['status']=='geographic_position_estimate'
    latlon=r['real_map']['target_geojson']['geometry']['coordinates']
    assert -180<=latlon[0]<=180 and -90<=latlon[1]<=90


def test_no_geo_from_valid_depth_without_site():
    r=project_dual_maps(obs(),intr(),pose(),depth(),None)
    assert r['real_map']['target_geojson'] is None
    assert r['status']=='relative_position_only'


def test_non_model_output_prevents_fabricated_position():
    r=project_dual_maps(obs(),intr(),pose(),replace(depth(),method='OTHER'),site())
    assert r['status']=='bearing_only'
    assert r['real_map']['target_geojson'] is None


def test_no_fake_95_percent_claim_in_output():
    r=project_dual_maps(obs(),intr(),pose(),depth(),site())
    assert '95%' not in str(r)
    assert 'DEPTH_INTERVAL_ONLY' in r['uncertainty_scope']