"""Camera-calibrated bearing and bounded map representations.

Never reconstruct global GPS or metric distance from an uncalibrated monocular image.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    calibration_id: str

    def __post_init__(self):
        if not (self.fx>0 and self.fy>0 and self.calibration_id and
                all(math.isfinite(v) for v in (self.fx,self.fy,self.cx,self.cy))):
            raise ValueError('VALID_INTRINSICS_REQUIRED')


def bearing_from_box(box, calibration: Intrinsics, bearing_uncertainty_deg: float | None=None):
    """Camera-relative azimuth/elevation only; no position or distance claims.

    Use undistorted pixel coordinates. Camera pose/extrinsics are *not* assumed.
    """
    x1,y1,x2,y2=map(float,box)
    if not all(math.isfinite(z) for z in (x1,y1,x2,y2)) or not (x1<x2 and y1<y2):
        raise ValueError('INVALID_BOX')
    if bearing_uncertainty_deg is not None and (not math.isfinite(bearing_uncertainty_deg) or bearing_uncertainty_deg<0):
        raise ValueError('INVALID_UNCERTAINTY')
    u=(x1+x2)/2; v=(y1+y2)/2
    return dict(azimuth_deg=math.degrees(math.atan2(u-calibration.cx,calibration.fx)),
                elevation_deg=-math.degrees(math.atan2(v-calibration.cy,calibration.fy)),
                calibration_id=calibration.calibration_id,
                uncertainty_deg=bearing_uncertainty_deg,
                coordinate_system='camera_relative_undistorted',
                range_m=None,latitude=None,longitude=None,
                provenance='calibrated_intrinsics; undistortion and pose verification required')


def synchronized_views(session_id: str, frame_index: int, detections: list[dict], bearings: list[dict]):
    """Both views carry identical object identities; geographic map never fabricates fixes."""
    if len(detections)!=len(bearings):
        raise ValueError('BEARING_DETECTION_COUNT_MISMATCH')
    objects=[]
    for det,bearing in zip(detections,bearings):
        if det.get('class_name')=='UNKNOWN' and det.get('track_id') is not None:
            raise ValueError('UNKNOWN_MUST_NOT_HAVE_ID')
        objects.append(dict(class_name=det['class_name'],track_id=det.get('track_id'),
                            box_xyxy=det['xyxy'],camera_bearing=bearing,
                            global_position_status='UNAVAILABLE_NO_GEOREFERENCED_MEASUREMENT'))
    shared=dict(session_id=session_id,frame_index=frame_index,objects=objects)
    return dict(tactical=dict(shared,view='tactical_camera_relative'),
                geographic=dict(shared,view='geographic',positions_available=False,
                                reason='No calibrated global geo-fix supplied'))
