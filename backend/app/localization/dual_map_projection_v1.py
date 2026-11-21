"""SkyGuard FUSION 42.6C: one evidence-gated position for two map views.

A pixel and bearing do NOT establish range or geographic position. The module
accepts 42.6B optical-axis depth ONLY when independently evidenced size,
calibration, and image measurement were declared upstream. An independently
surveyed camera pose and a measured site reference are required to project a
real-map point. No basemap tiles, network, camera or model are touched here.

Output intervals apply to the single optical-axis depth input only: camera
intrinsics, pose, pixel center and survey errors are NOT included. They must
not be misrepresented as total geolocation confidence or 95% probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, hypot, isfinite, radians, sin, sqrt
from typing import Optional, Sequence

from app.localization.range_evidence_guard_v1 import RangeAssessment

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_B = WGS84_A * (1.0 - WGS84_F)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _source(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _three(values: object) -> bool:
    return isinstance(values, (tuple, list)) and len(values) == 3 and all(_finite_number(v) for v in values)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _length(v: Sequence[float]) -> float:
    return sqrt(_dot(v, v))


def _valid_rotation(matrix: object) -> bool:
    """Camera axes (right, down, forward) to local ENU, no reflections/scales."""
    if not isinstance(matrix, (tuple, list)) or len(matrix) != 3:
        return False
    if not all(_three(row) for row in matrix):
        return False
    rows = matrix
    if any(abs(_dot(row, row) - 1) > 1e-5 for row in rows):
        return False
    if any(abs(_dot(rows[i], rows[j])) > 1e-5 for i in range(3) for j in range(i + 1, 3)):
        return False
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    return abs(determinant - 1) <= 1e-5


@dataclass(frozen=True)
class CameraIntrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    image_width_px: int
    image_height_px: int
    calibration_source: str
    verified: bool


@dataclass(frozen=True)
class SurveyedCameraPose:
    camera_east_m: float
    camera_north_m: float
    camera_up_m: float
    # 3x3 right-handed rotation, CV camera frame (+x right,+y down,+z forward) -> ENU.
    camera_to_enu: tuple[tuple[float, float, float], ...]
    survey_source: str
    orientation_source: str
    verified: bool


@dataclass(frozen=True)
class SiteWGS84Reference:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    facility_id: str
    survey_source: str
    verified: bool


@dataclass(frozen=True)
class TrackFrameObservation:
    session_id: str
    frame_id: str
    frame_index: int
    timestamp_ms: int
    track_id: str
    object_class: str
    confidence: float
    center_x_px: float
    center_y_px: float


def _valid_observation(o: object) -> bool:
    return (
        isinstance(o, TrackFrameObservation)
        and all(_source(x) for x in (o.session_id, o.frame_id, o.track_id, o.object_class))
        and type(o.frame_index) is int and o.frame_index >= 0
        and type(o.timestamp_ms) is int and o.timestamp_ms >= 0
        and _finite_number(o.confidence) and 0 <= o.confidence <= 1
        and _finite_number(o.center_x_px) and _finite_number(o.center_y_px)
    )


def _valid_intrinsics(i: object) -> bool:
    return (
        isinstance(i, CameraIntrinsics) and i.verified is True and _source(i.calibration_source)
        and all(_finite_number(v) for v in (i.fx_px, i.fy_px, i.cx_px, i.cy_px))
        and i.fx_px > 0 and i.fy_px > 0
        and type(i.image_width_px) is int and type(i.image_height_px) is int
        and i.image_width_px > 0 and i.image_height_px > 0
        and 0 <= i.cx_px < i.image_width_px and 0 <= i.cy_px < i.image_height_px
    )


def _valid_pose(p: object) -> bool:
    return (
        isinstance(p, SurveyedCameraPose) and p.verified is True
        and _source(p.survey_source) and _source(p.orientation_source)
        and all(_finite_number(v) for v in (p.camera_east_m, p.camera_north_m, p.camera_up_m))
        and _valid_rotation(p.camera_to_enu)
    )


def _valid_site(s: object) -> bool:
    return (
        isinstance(s, SiteWGS84Reference) and s.verified is True
        and _source(s.facility_id) and _source(s.survey_source)
        and all(_finite_number(v) for v in (s.latitude_deg, s.longitude_deg, s.altitude_m))
        and -90 <= s.latitude_deg <= 90 and -180 <= s.longitude_deg <= 180
    )


def _valid_depth(r: object, o: TrackFrameObservation) -> bool:
    if not isinstance(r, RangeAssessment) or r.status != 'estimated':
        return False
    if r.frame_id != o.frame_id or r.target_track_id != o.track_id:
        return False
    if r.method != 'CALIBRATED_MONOCULAR_APPARENT_SIZE_PINHOLE':
        return False
    values = (r.camera_optical_axis_depth_m, r.conservative_lower_m, r.conservative_upper_m)
    return (all(_finite_number(v) and v > 0 for v in values)
            and values[1] <= values[0] <= values[2]
            and r.facility_distance_m is None and r.latitude is None and r.longitude is None)


def _wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    lat, lon = radians(lat_deg), radians(lon_deg)
    sl, cl = sin(lat), cos(lat)
    n = WGS84_A / sqrt(1 - WGS84_E2 * sl * sl)
    return ((n + alt_m) * cl * cos(lon), (n + alt_m) * cl * sin(lon),
            (n * (1 - WGS84_E2) + alt_m) * sl)


def _enu_to_wgs84(site: SiteWGS84Reference, east: float, north: float, up: float) -> tuple[float, float, float]:
    lat, lon = radians(site.latitude_deg), radians(site.longitude_deg)
    origin = _wgs84_to_ecef(site.latitude_deg, site.longitude_deg, site.altitude_m)
    x = origin[0] - sin(lon) * east - sin(lat) * cos(lon) * north + cos(lat) * cos(lon) * up
    y = origin[1] + cos(lon) * east - sin(lat) * sin(lon) * north + cos(lat) * sin(lon) * up
    z = origin[2] + cos(lat) * north + sin(lat) * up
    radial = hypot(x, y)
    lon_out = degrees(atan2(y, x)) if radial > 1e-8 else site.longitude_deg
    if radial < 1e-8:
        return (90.0 if z >= 0 else -90.0, lon_out, abs(z) - WGS84_B)
    lat_out = atan2(z, radial * (1 - WGS84_E2))
    for _ in range(12):
        normal = WGS84_A / sqrt(1 - WGS84_E2 * sin(lat_out) ** 2)
        lat_new = atan2(z + WGS84_E2 * normal * sin(lat_out), radial)
        if abs(lat_new - lat_out) < 1e-14:
            lat_out = lat_new
            break
        lat_out = lat_new
    normal = WGS84_A / sqrt(1 - WGS84_E2 * sin(lat_out) ** 2)
    altitude = radial / cos(lat_out) - normal
    return (degrees(lat_out), lon_out, altitude)


def _geojson_point(lon_deg: float, lat_deg: float, altitude_m: float, props: dict) -> dict:
    return {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [lon_deg, lat_deg, altitude_m]},
            'properties': dict(props)}


def project_dual_maps(
    observation: TrackFrameObservation | None,
    intrinsics: CameraIntrinsics | None,
    pose: SurveyedCameraPose | None,
    range_assessment: Optional[RangeAssessment] = None,
    site: Optional[SiteWGS84Reference] = None,
) -> dict:
    """Return matching tactical/real-map payloads; NEVER invent a point for bearing-only.

    `range_assessment` is camera optical-axis depth, not slant/facility range.
    Uncertainty bounds are depth-interval-only, NOT full physical accuracy.
    """
    identity = {
        'session_id': getattr(observation, 'session_id', None),
        'frame_id': getattr(observation, 'frame_id', None),
        'frame_index': getattr(observation, 'frame_index', None),
        'timestamp_ms': getattr(observation, 'timestamp_ms', None),
        'track_id': getattr(observation, 'track_id', None),
    }
    def unknown(reason: str) -> dict:
        return {'schema_version': '42.6C-v1', **identity, 'status': 'unknown',
                'reason': reason, 'tactical_map': None, 'real_map': None,
                'uncertainty_scope': 'no_position_evidence',
                'field_validated': False}
    if not _valid_observation(observation):
        return unknown('INVALID_TRACK_FRAME_OBSERVATION')
    if not _valid_intrinsics(intrinsics):
        return unknown('UNVERIFIED_CAMERA_INTRINSICS')
    if not (0 <= observation.center_x_px < intrinsics.image_width_px and
            0 <= observation.center_y_px < intrinsics.image_height_px):
        return unknown('PIXEL_OUTSIDE_CALIBRATED_FRAME')
    if not _valid_pose(pose):
        return unknown('UNVERIFIED_CAMERA_POSE_OR_ORIENTATION')

    xn = (observation.center_x_px - intrinsics.cx_px) / intrinsics.fx_px
    yn = (observation.center_y_px - intrinsics.cy_px) / intrinsics.fy_px
    ray_cv = (xn, yn, 1.0)
    ray_enu = tuple(_dot(row, ray_cv) for row in pose.camera_to_enu)
    length = _length(ray_enu)
    if not isfinite(length) or length <= 0:
        return unknown('NUMERIC_UNSTABLE_RAY')
    direction = tuple(v / length for v in ray_enu)
    # Bearing is clockwise from north, with no assumed target range.
    bearing = (degrees(atan2(direction[0], direction[1])) + 360.0) % 360.0
    if not all(isfinite(v) for v in direction):
        return unknown('NUMERIC_UNSTABLE_RAY')
    common = {**identity, 'class': observation.object_class,
              'confidence': observation.confidence}
    cam_enu = [pose.camera_east_m, pose.camera_north_m, pose.camera_up_m]
    tactical = {**common, 'crs': 'LOCAL_ENU_METRES_FROM_FACILITY',
                'camera_enu_m': cam_enu, 'position_enu_m': None,
                'bearing_deg_from_true_north': bearing,
                'ray_unit_enu': list(direction),
                'facility_distance_m': None, 'facility_distance_depth_bounds_m': None,
                'bound_scope': None}
    real = {**common, 'crs': 'WGS84_EPSG_4979', 'target_geojson': None,
            'camera_geojson': None, 'bearing_deg_from_true_north': bearing,
            'basemap_layer': 'EXTERNAL_LICENSED_PROVIDER_NOT_IMPLEMENTED'}

    has_site = _valid_site(site)
    if has_site:
        cam_lat, cam_lon, cam_alt = _enu_to_wgs84(site, *cam_enu)
        real['camera_geojson'] = _geojson_point(cam_lon, cam_lat, cam_alt, {
            'role': 'camera', 'facility_id': site.facility_id, 'source': site.survey_source})

    if not _valid_depth(range_assessment, observation):
        reason = 'MISSING_OR_UNVERIFIED_OPTICAL_DEPTH'
        if isinstance(range_assessment, RangeAssessment) and range_assessment.status == 'estimated':
            if range_assessment.frame_id != observation.frame_id or range_assessment.target_track_id != observation.track_id:
                reason = 'STALE_OR_FOREIGN_RANGE_EVIDENCE'
            else:
                reason = 'INVALID_RANGE_INTERVAL_OR_METHOD'
        return {'schema_version': '42.6C-v1', **identity, 'status': 'bearing_only',
                'reason': reason, 'tactical_map': tactical, 'real_map': real,
                'uncertainty_scope': 'direction_only_intrinsics_pose_not_error_bounded',
                'field_validated': False}

    depth = range_assessment.camera_optical_axis_depth_m
    lo, hi = range_assessment.conservative_lower_m, range_assessment.conservative_upper_m
    position = [cam_enu[k] + depth * ray_enu[k] for k in range(3)]
    if not all(isfinite(v) for v in position):
        return unknown('NUMERIC_UNSTABLE_POSITION')
    # Facility radial distance may not be monotonic in depth when camera is offset;
    # minimize quadratic distance along the entire bounded ray interval.
    a = _dot(ray_enu, ray_enu)
    b = _dot(cam_enu, ray_enu)
    candidate_min_depth = min(hi, max(lo, -b / a))
    distance_at = lambda z: _length([cam_enu[k] + z * ray_enu[k] for k in range(3)])
    distance = _length(position)
    lower = min(distance_at(lo), distance_at(hi), distance_at(candidate_min_depth))
    upper = max(distance_at(lo), distance_at(hi))
    if not all(isfinite(v) for v in (distance, lower, upper)) or lower > distance or distance > upper:
        return unknown('NUMERIC_UNSTABLE_DISTANCE')
    tactical['position_enu_m'] = position
    tactical['facility_distance_m'] = distance
    tactical['facility_distance_depth_bounds_m'] = [lower, upper]
    tactical['bound_scope'] = 'DEPTH_INTERVAL_ONLY_NOT_FULL_UNCERTAINTY'
    real['target_geojson'] = None
    result_status = 'relative_position_only'
    result_reason = 'SITE_WGS84_NOT_INDEPENDENTLY_VERIFIED'
    if has_site:
        lat, lon, alt = _enu_to_wgs84(site, *position)
        if not all(isfinite(v) for v in (lat, lon, alt)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return unknown('NUMERIC_UNSTABLE_GEOLOCATION')
        real['target_geojson'] = _geojson_point(lon, lat, alt, {
            'role': 'estimated_target', **common, 'method': range_assessment.method,
            'uncertainty_scope': 'DEPTH_INTERVAL_ONLY_NOT_FULL_UNCERTAINTY',
            'field_validated': False})
        result_status = 'geographic_position_estimate'
        result_reason = 'EVIDENCE_GATED_GEOMETRY_NOT_FIELD_VALIDATED'
    return {'schema_version': '42.6C-v1', **identity,
            'status': result_status, 'reason': result_reason,
            'tactical_map': tactical, 'real_map': real,
            'uncertainty_scope': 'DEPTH_INTERVAL_ONLY_NOT_FULL_UNCERTAINTY',
            'field_validated': False}