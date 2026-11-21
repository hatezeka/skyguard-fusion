"""Single scene contract serving tactical and real geographic views.

No monocular range or GPS inference. Ground position requires independently
measured slant/horizontal range + verified camera georeference. With absent
range/pose evidence the point is null, while calibrated bearing is optional.
"""
from __future__ import annotations
import math


def finite_number(value, low=None, high=None) -> float:
    val = float(value)
    if not math.isfinite(val) or (low is not None and val < low) or (high is not None and val > high):
        raise ValueError('Invalid numeric evidence')
    return val


def pixel_bearing(box: list[float], width: int, hfov_deg: float,
                  azimuth_deg: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    hfov = finite_number(hfov_deg, 0.001, 179.999)
    if not 0 <= finite_number(box[0]) <= width or box[2] <= 0:
        raise ValueError('bbox out of image')
    cx = float(box[0]) + float(box[2])/2
    if cx < 0 or cx > width:
        raise ValueError('bbox center out of image')
    x = (2 * cx/width - 1.) * math.tan(math.radians(hfov/2))
    return (finite_number(azimuth_deg) + math.degrees(math.atan(x))) % 360


def point_from_verified_range(camera: dict | None, bearing_deg: float | None,
                              measurement: dict | None) -> dict:
    """Only accepts externally measured horizontal range with confidence bounds."""
    none = {'east_m': None, 'north_m': None, 'wgs84': None,
            'horizontal_range_m': None, 'uncertainty_m': None,
            'status': 'BEARING_ONLY' if bearing_deg is not None else 'NO_POSITION_EVIDENCE'}
    if measurement is None or bearing_deg is None:
        return none
    if measurement.get('source') not in ('SURVEY', 'RTK', 'RANGEFINDER', 'CALIBRATED_STEREO'):
        raise ValueError('Unverified range provenance')
    horizontal = finite_number(measurement.get('horizontal_range_m'), 0)
    sigma = finite_number(measurement.get('uncertainty_m'), 0)
    if camera is None or not camera.get('georeference_verified', False):
        return {**none, 'status': 'RANGE_ONLY_NO_VERIFIED_GEO'}
    lat = finite_number(camera.get('latitude_deg'), -90, 90)
    lon = finite_number(camera.get('longitude_deg'), -180, 180)
    heading = math.radians(finite_number(bearing_deg) % 360)
    east, north = horizontal * math.sin(heading), horizontal * math.cos(heading)
    # Spherical WGS84-like approximation, labelled approximate (not survey grade).
    angular = horizontal/6371008.8
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(angular)+math.cos(lat1)*math.sin(angular)*math.cos(heading))
    lon2 = lon1 + math.atan2(math.sin(heading)*math.sin(angular)*math.cos(lat1),
                            math.cos(angular)-math.sin(lat1)*math.sin(lat2))
    return {'east_m': east, 'north_m': north,
            'wgs84': {'latitude_deg': math.degrees(lat2),
                      'longitude_deg': ((math.degrees(lon2)+180)%360)-180,
                      'method': 'APPROXIMATE_GEODESIC_WITH_VERIFIED_EXTERNAL_RANGE'},
            'horizontal_range_m': horizontal, 'uncertainty_m': sigma,
            'status': 'EXTERNAL_RANGE_VERIFIED'}


def dual_scene(session_id: str, frame_index: int, timestamp_ms: float,
               tracks: list[dict], width: int, *, hfov_deg: float | None = None,
               camera: dict | None = None,
               measured_ranges: dict[int, dict] | None = None) -> dict:
    """Same objects/IDs/source frame feed both maps; absent data represented null."""
    measured_ranges = measured_ranges or {}
    geo, tactical = [], []
    for t in tracks:
        cls = str(t['class']).upper()
        if cls not in ('DRONE', 'BIRD'):
            raise ValueError('UNKNOWN/non-target cannot be mapped as tracked target')
        tid = int(t['track_id'])
        bearing = None
        if hfov_deg is not None and camera is not None and 'azimuth_deg' in camera:
            bearing = pixel_bearing(t['bbox'], width, hfov_deg, camera['azimuth_deg'])
        pos = point_from_verified_range(camera, bearing, measured_ranges.get(tid))
        shared = {'track_id': tid, 'class': cls, 'session_id': str(session_id),
                  'frame_index': int(frame_index), 'timestamp_ms': float(timestamp_ms),
                  'bearing_deg': bearing, 'uncertainty_m': pos['uncertainty_m'],
                  'location_status': pos['status']}
        tactical.append({**shared, 'bbox_px': t['bbox'],
                         'east_m': pos['east_m'], 'north_m': pos['north_m']})
        geo.append({**shared, 'wgs84': pos['wgs84']})
    return {'contract_version': 'skyguard-dual-map-v1',
            'session_id': str(session_id), 'frame_index': int(frame_index),
            'timestamp_ms': float(timestamp_ms),
            'tactical_map': {'objects': tactical}, 'real_map': {'objects': geo},
            'basemap': {'status': 'USE_EXISTING_LICENSED_FRONTEND_PROVIDER',
                        'coordinates_invented': False}}
