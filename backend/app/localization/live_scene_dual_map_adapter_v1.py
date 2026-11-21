"""42.6D: join existing protected live-map scene to the common dual-map schema.

No new endpoint or camera access. Existing manually supplied camera coordinates
and display-only bearing guides are NEVER promoted into target GPS/range.

TrustedTrackEvidence is intended only for internal server-side ingestion once a
frame-pinned tracking/evidence pipeline exists. It is NOT a public request model.
Current live/map/scene has no source of such evidence; its default results are
necessarily unverified directions with no inferred target position.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Optional, Sequence

from app.localization.dual_map_projection_v1 import (
    CameraIntrinsics, SiteWGS84Reference, SurveyedCameraPose,
    TrackFrameObservation, project_dual_maps,
)
from app.localization.range_evidence_guard_v1 import RangeAssessment


@dataclass(frozen=True)
class TrustedTrackEvidence:
    """Explicit server-internal evidence, NEVER accepted directly from browser."""
    detection_index: int
    observation: TrackFrameObservation
    intrinsics: CameraIntrinsics
    pose: SurveyedCameraPose
    range_assessment: Optional[RangeAssessment] = None
    site: Optional[SiteWGS84Reference] = None


def _valid_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _check_scene(scene: Mapping) -> None:
    if not isinstance(scene, Mapping) or scene.get('scene_type') != 'live_camera_bearing_map':
        raise ValueError('EXPECTED_EXISTING_LIVE_MAP_SCENE')
    if not isinstance(scene.get('camera_id'), str) or not scene['camera_id']:
        raise ValueError('INVALID_CAMERA_ID')
    if not isinstance(scene.get('session_id'), str) or not scene['session_id']:
        raise ValueError('INVALID_SESSION_ID')
    if not _valid_int(scene.get('frame_index')):
        raise ValueError('INVALID_FRAME_INDEX')
    observations = scene.get('observations')
    if not isinstance(observations, list) or len(observations) > 512:
        raise ValueError('INVALID_LIVE_OBSERVATIONS')
    if not isinstance(scene.get('geojson'), Mapping):
        raise ValueError('MISSING_LEGACY_GEOJSON')
    if scene.get('geojson', {}).get('type') != 'FeatureCollection':
        raise ValueError('INVALID_LEGACY_GEOJSON')
    for obs in observations:
        if not isinstance(obs, Mapping) or not _valid_int(obs.get('detection_index')):
            raise ValueError('INVALID_DETECTION_INDEX')
        if obs.get('object_class') not in ('drone', 'bird', 'unknown'):
            raise ValueError('INVALID_DETECTION_CLASS')
        conf = obs.get('confidence')
        if type(conf) not in (float, int) or not isfinite(conf) or not (0 <= conf <= 1):
            raise ValueError('INVALID_DETECTION_CONFIDENCE')


def _verify_bound_evidence(scene: Mapping, source: Mapping, record: TrustedTrackEvidence) -> None:
    """Fail closed: frame+session+timestamp+stable ID must all agree."""
    o = record.observation
    if not isinstance(o, TrackFrameObservation) or not _valid_int(record.detection_index):
        raise ValueError('INVALID_TRUSTED_EVIDENCE')
    if (o.session_id != scene['session_id'] or o.frame_index != scene['frame_index']
        or not isinstance(scene.get('frame_id'), str) or o.frame_id != scene['frame_id']
        or not _valid_int(scene.get('timestamp_ms')) or o.timestamp_ms != scene['timestamp_ms']):
        raise ValueError('STALE_OR_UNBOUND_TRACK_FRAME_EVIDENCE')
    if (not isinstance(source.get('track_id'), str) or not source['track_id']
        or o.track_id != source['track_id']):
        raise ValueError('TRACK_ID_NOT_BOUND_TO_LIVE_OBSERVATION')
    if o.object_class != source['object_class']:
        raise ValueError('TRACK_CLASS_MISMATCH')
    if abs(o.confidence - source['confidence']) > 1e-6:
        raise ValueError('TRACK_CONFIDENCE_MISMATCH')


def attach_dual_map_state(
    scene: Mapping,
    *,
    trusted_evidence: Sequence[TrustedTrackEvidence] = (),
) -> dict:
    """Add ``dual_map_v1`` to the existing scene without changing legacy fields.

    Default path: current production scene lacks secure range/pose/time evidence.
    Hence output contains only unverified direction, no target GPS or metres.
    When trusted evidence becomes available INSIDE the server pipeline, all
    frame/identity fields must match before the 42.6C projector is consulted.
    """
    _check_scene(scene)
    if not isinstance(trusted_evidence, (tuple, list)) or len(trusted_evidence) > 512:
        raise ValueError('INVALID_TRUSTED_EVIDENCE_COLLECTION')
    items = scene['observations']
    by_index = {}
    for source in items:
        idx = source['detection_index']
        if idx in by_index:
            raise ValueError('DUPLICATE_DETECTION_INDEX')
        by_index[idx] = source
    records = {}
    for record in trusted_evidence:
        if not isinstance(record, TrustedTrackEvidence):
            raise ValueError('INVALID_TRUSTED_EVIDENCE_RECORD')
        idx = record.detection_index
        if idx not in by_index or idx in records:
            raise ValueError('UNKNOWN_OR_DUPLICATE_TRUSTED_DETECTION_INDEX')
        _verify_bound_evidence(scene, by_index[idx], record)
        records[idx] = record

    tactical_objects = []
    geo_features = []
    any_position = False
    any_geo = False
    for source in items:
        idx = source['detection_index']
        track_id = source.get('track_id')
        if not isinstance(track_id, str) or not track_id:
            track_id = None
        common = {
            'detection_index': idx, 'track_id': track_id,
            'track_identity_verified': False if idx not in records else bool(track_id),
            'object_class': source['object_class'], 'confidence': source['confidence'],
            'frame_index': scene['frame_index'], 'session_id': scene['session_id'],
            'bearing_deg_declared': source.get('azimuth_degrees'),
            'position_enu_m': None, 'facility_distance_m': None,
            'facility_distance_depth_bounds_m': None,
            'target_geojson': None,
            'status': 'unverified_direction_only',
            'uncertainty_scope': 'orientation_not_physically_verified',
        }
        if idx in records:
            record = records[idx]
            result = project_dual_maps(
                record.observation, record.intrinsics, record.pose,
                record.range_assessment, record.site,
            )
            common['status'] = result['status']
            common['uncertainty_scope'] = result['uncertainty_scope']
            tactical = result.get('tactical_map')
            geo = result.get('real_map')
            if tactical:
                common['position_enu_m'] = tactical.get('position_enu_m')
                common['facility_distance_m'] = tactical.get('facility_distance_m')
                common['facility_distance_depth_bounds_m'] = tactical.get('facility_distance_depth_bounds_m')
                if common['position_enu_m'] is not None:
                    any_position = True
            if geo and geo.get('target_geojson') is not None:
                common['target_geojson'] = geo['target_geojson']
                geo_features.append(geo['target_geojson'])
                any_geo = True
        tactical_objects.append(common)

    output = deepcopy(dict(scene))
    output['dual_map_v1'] = {
        'schema_version': '42.6D-v1', 'camera_id': scene['camera_id'],
        'session_id': scene['session_id'], 'frame_index': scene['frame_index'],
        'frame_id': scene.get('frame_id'), 'timestamp_ms': scene.get('timestamp_ms'),
        'tactical_map': {'crs': 'LOCAL_ENU_METRES_FROM_FACILITY',
                         'objects': tactical_objects},
        'real_map': {'crs': 'WGS84_EPSG_4979',
                     'verified_estimate_geojson': {
                         'type': 'FeatureCollection', 'features': geo_features}},
        'legacy_display_geojson': deepcopy(scene['geojson']),
        'position_estimate_present': any_position,
        'geographic_estimate_present': any_geo,
        'physical_validation_completed': False,
        'notes': 'Legacy camera points and bearing guides are display-only; no inferred target coordinates',
    }
    # Maintain legacy semantics if future INTERNAL pipeline supplies verified data.
    if any_geo:
        output['target_gps_available'] = True
    if any_position:
        output['target_range_available'] = True
    return output