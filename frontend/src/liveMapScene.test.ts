import { describe, expect, it } from 'vitest'
import { parseLiveMapScene, type LiveMapSceneRequest } from './api/liveMapScene'

const ID = '0123456789abcdef0123456789abcdef'
const HASH = 'a'.repeat(64)
const req: LiveMapSceneRequest = {
  camera_id: 'cam-A', expected_session_id: ID, expected_frame_index: 17,
  north_reference: 'true_north',
}
const base = () => ({
  scene_type: 'live_camera_bearing_map', camera_id: 'cam-A', session_id: ID,
  frame_index: 17, north_reference: 'true_north', calibration_snapshot_sha256: HASH,
  session_identity_verified: true, calibration_bound_to_session_verified: true,
  calibration_bound_to_frame_verified: true, physical_calibration_verified: false,
  camera_location_bound_to_session_verified: false, camera_location_survey_verified: false,
  camera_location_source: 'manual', camera_location: {
    latitude: 42.7, longitude: 21.1, source: 'manual',
    horizontal_accuracy_m: null, captured_at: null,
    survey_verified: false, bound_to_session_verified: false,
  },
  target_gps_available: false, target_range_available: false,
  bearing_guides_are_display_only: true, magnetic_declination_applied: false,
  observations: [{
    detection_index: 0, object_class: 'drone', confidence: .8,
    azimuth_degrees: 80, elevation_degrees: 12, north_reference: 'true_north',
    target_location: null, target_range_m: null, target_location_observable: false,
  }],
  geojson: { type: 'FeatureCollection', features: [
    { type: 'Feature', geometry: { type: 'Point', coordinates: [21.1, 42.7] }, properties: {
      kind: 'camera', camera_id: 'cam-A', location_source: 'manual',
      survey_verified: false, location_bound_to_session_verified: false,
    } },
    { type: 'Feature', geometry: { type: 'LineString', coordinates: [[21.1, 42.7], [21.10244, 42.7]] }, properties: {
      kind: 'bearing_guide', detection_index: 0, object_class: 'drone',
      azimuth_degrees: 80, display_only: true, guide_length_m: 200,
      is_target_position: false, is_measured_range: false, north_reference: 'true_north',
    } },
  ] },
})

describe('Phase 40.2 scene validation', () => {
  it('accepts only a matching protected scene', () => {
    const scene = parseLiveMapScene(base(), req)
    expect(scene.observations).toHaveLength(1)
    expect(scene.target_gps_available).toBe(false)
  })
  it('rejects a stale session and frame', () => {
    expect(() => parseLiveMapScene({ ...base(), session_id: 'f'.repeat(32) }, req)).toThrow()
    expect(() => parseLiveMapScene({ ...base(), frame_index: 18 }, req)).toThrow()
  })
  it('rejects claims of GPS or measured range', () => {
    expect(() => parseLiveMapScene({ ...base(), target_gps_available: true }, req)).toThrow()
    expect(() => parseLiveMapScene({ ...base(), target_range_available: true }, req)).toThrow()
  })
  it('rejects guide masquerading as a target location', () => {
    const value = base();
    (value.geojson.features[1].properties as { is_target_position: boolean }).is_target_position = true
    expect(() => parseLiveMapScene(value, req)).toThrow()
  })
  it('rejects cross-camera and duplicate marker', () => {
    const value = base()
    value.geojson.features.push(value.geojson.features[0])
    expect(() => parseLiveMapScene(value, req)).toThrow()
    expect(() => parseLiveMapScene({ ...base(), camera_id: 'different' }, req)).toThrow()
  })
  it('accepts no-camera-location without invented features', () => {
    const value = { ...base(), camera_location: null, camera_location_source: 'unavailable',
      geojson: { type: 'FeatureCollection', features: [] } }
    const scene = parseLiveMapScene(value, req)
    expect(scene.camera_location).toBeNull()
  })
  it('suppresses geographic guides for magnetic north', () => {
    const value = { ...base(), north_reference: 'magnetic_north' }
    const magneticReq: LiveMapSceneRequest = { ...req, north_reference: 'magnetic_north' }
    expect(() => parseLiveMapScene(value, magneticReq)).toThrow()
  })
  it('rejects physically verified claim in synthetic scene', () => {
    expect(() => parseLiveMapScene({ ...base(), physical_calibration_verified: true }, req)).toThrow()
  })
})
