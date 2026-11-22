import pytest
from skyguard_acceptance.geometry import Intrinsics,bearing_from_box,synchronized_views
from skyguard_acceptance.criteria import acceptance_matrix,REQUIRED

def test_camera_center_has_zero_bearing_and_no_invented_range():
    c=Intrinsics(1000,1000,640,360,'camera-cal-01')
    r=bearing_from_box((630,350,650,370),c)
    assert r['azimuth_deg']==0 and r['elevation_deg']==0
    assert r['range_m'] is None and r['latitude'] is None

def test_reject_bad_intrinsics():
    with pytest.raises(ValueError): Intrinsics(0,1000,640,360,'x')

def test_unknown_cannot_be_tracked():
    with pytest.raises(ValueError): synchronized_views('s',0,[dict(class_name='UNKNOWN',track_id=2,xyxy=(0,0,5,5))],[{}])

def test_both_views_same_objects_without_gps():
    d=[dict(class_name='DRONE',track_id=7,xyxy=(2,3,10,12))]
    views=synchronized_views('s',1,d,[{'azimuth_deg':1}])
    assert views['tactical']['objects']==views['geographic']['objects']
    assert not views['geographic']['positions_available']

def test_acceptance_fails_closed():
    result=acceptance_matrix({'trained_checkpoint':{'status':'PASS','evidence':['sha256:abc']}})
    assert result['overall']=='EVIDENCE_INCOMPLETE'
    assert result['criteria']['physical_usb_drone']['status']=='NOT_TESTED'

def test_no_unsupported_pass():
    with pytest.raises(ValueError): acceptance_matrix({'trained_checkpoint':{'status':'PASS'}})
