/** Phase 40.2: explicitly unlocated targets. GeoJSON lines are display guides only. */
export type NorthReference = 'true_north' | 'magnetic_north'

export interface LiveMapSceneRequest {
  camera_id: string
  expected_session_id: string
  expected_frame_index: number
  north_reference: NorthReference
}

type Coordinate = [number, number] // GeoJSON [longitude, latitude]

export interface CameraMapLocation {
  latitude: number
  longitude: number
  source: string
  horizontal_accuracy_m: number | null
  captured_at: string | null
  survey_verified: false
  bound_to_session_verified: false
}

export interface LiveMapObservation {
  detection_index: number
  object_class: 'drone' | 'bird' | 'unknown'
  confidence: number
  azimuth_degrees: number | null
  elevation_degrees: number
  north_reference: NorthReference
  target_location: null
  target_range_m: null
  target_location_observable: false
}

export interface CameraGeoFeature {
  type: 'Feature'
  geometry: { type: 'Point'; coordinates: Coordinate }
  properties: {
    kind: 'camera'
    camera_id: string
    location_source: string
    survey_verified: false
    location_bound_to_session_verified: false
  }
}

export interface GuideGeoFeature {
  type: 'Feature'
  geometry: { type: 'LineString'; coordinates: [Coordinate, Coordinate] }
  properties: {
    kind: 'bearing_guide'
    detection_index: number
    object_class: string
    azimuth_degrees: number
    display_only: true
    guide_length_m: number
    is_target_position: false
    is_measured_range: false
    north_reference: 'true_north'
  }
}

export interface LiveMapScene {
  scene_type: 'live_camera_bearing_map'
  camera_id: string
  session_id: string
  frame_index: number
  north_reference: NorthReference
  calibration_snapshot_sha256: string
  session_identity_verified: true
  calibration_bound_to_session_verified: true
  calibration_bound_to_frame_verified: true
  physical_calibration_verified: false
  camera_location_bound_to_session_verified: false
  camera_location_survey_verified: false
  camera_location_source: string
  camera_location: CameraMapLocation | null
  target_gps_available: false
  target_range_available: false
  bearing_guides_are_display_only: true
  magnetic_declination_applied: false
  observations: LiveMapObservation[]
  geojson: { type: 'FeatureCollection'; features: Array<CameraGeoFeature | GuideGeoFeature> }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
function finite(value: unknown, low: number, high: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= low && value <= high
}
function coordinate(value: unknown): value is Coordinate {
  return Array.isArray(value) && value.length === 2 && finite(value[0], -180, 180) && finite(value[1], -90, 90)
}
function equalCoords(a: Coordinate, b: Coordinate): boolean {
  return a[0] === b[0] && a[1] === b[1]
}

function guideDistanceMeters(a: Coordinate, b: Coordinate): number {
  const rad = Math.PI / 180
  const lat1 = a[1] * rad
  const lat2 = b[1] * rad
  const dLat = (b[1] - a[1]) * rad
  const dLon = (b[0] - a[0]) * rad
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2
  return 2 * 6371008.8 * Math.asin(Math.min(1, Math.sqrt(h)))
}

/** Validation is fail-closed: a map must never invent target GPS or measured distance. */
export function parseLiveMapScene(data: unknown, request: LiveMapSceneRequest): LiveMapScene {
  if (!isRecord(data) || data.scene_type !== 'live_camera_bearing_map') throw Error('Invalid scene format')
  if (
    data.camera_id !== request.camera_id ||
    data.session_id !== request.expected_session_id ||
    data.frame_index !== request.expected_frame_index ||
    data.north_reference !== request.north_reference
  ) throw Error('Stale scene identity')
  if (
    data.session_identity_verified !== true ||
    data.calibration_bound_to_session_verified !== true ||
    data.calibration_bound_to_frame_verified !== true ||
    data.physical_calibration_verified !== false ||
    data.camera_location_bound_to_session_verified !== false ||
    data.camera_location_survey_verified !== false ||
    data.target_gps_available !== false ||
    data.target_range_available !== false ||
    data.bearing_guides_are_display_only !== true ||
    data.magnetic_declination_applied !== false ||
    typeof data.calibration_snapshot_sha256 !== 'string' ||
    !/^[0-9a-f]{64}$/.test(data.calibration_snapshot_sha256)
  ) throw Error('Unsafe map provenance')

  const location = data.camera_location
  let cameraCoordinate: Coordinate | null = null
  if (location !== null) {
    if (!isRecord(location) ||
      !finite(location.latitude, -90, 90) || !finite(location.longitude, -180, 180) ||
      location.survey_verified !== false || location.bound_to_session_verified !== false ||
      typeof location.source !== 'string' ||
      (location.horizontal_accuracy_m !== null && !finite(location.horizontal_accuracy_m, 0, 100000000)) ||
      (location.captured_at !== null && typeof location.captured_at !== 'string')
    ) throw Error('Invalid camera geolocation')
    cameraCoordinate = [location.longitude, location.latitude]
  }
  const collection = data.geojson
  if (!isRecord(collection) || collection.type !== 'FeatureCollection' ||
      !Array.isArray(collection.features) || collection.features.length > 513) throw Error('Invalid GeoJSON')

  let cameraCount = 0
  const guideIndexes = new Set<number>()
  for (const feature of collection.features) {
    if (!isRecord(feature) || feature.type !== 'Feature' ||
        !isRecord(feature.geometry) || !isRecord(feature.properties)) throw Error('Invalid feature')
    const props = feature.properties
    const geometry = feature.geometry
    if (props.kind === 'camera') {
      cameraCount += 1
      if (geometry.type !== 'Point' || !coordinate(geometry.coordinates) ||
          !cameraCoordinate || !equalCoords(geometry.coordinates, cameraCoordinate) ||
          props.camera_id !== request.camera_id || props.survey_verified !== false ||
          props.location_bound_to_session_verified !== false) throw Error('Invalid camera marker')
    } else if (props.kind === 'bearing_guide') {
      const coords = geometry.coordinates
      if (!cameraCoordinate || request.north_reference !== 'true_north' ||
          geometry.type !== 'LineString' || !Array.isArray(coords) || coords.length !== 2 ||
          !coordinate(coords[0]) || !coordinate(coords[1]) || !equalCoords(coords[0], cameraCoordinate) ||
          guideDistanceMeters(coords[0], coords[1]) < 195 ||
          guideDistanceMeters(coords[0], coords[1]) > 205 ||
          props.display_only !== true || props.is_target_position !== false ||
          props.is_measured_range !== false || props.north_reference !== 'true_north' ||
          props.guide_length_m !== 200 ||
          typeof props.detection_index !== 'number' || !Number.isInteger(props.detection_index) || props.detection_index < 0 ||
          guideIndexes.has(props.detection_index as number) ||
          !finite(props.azimuth_degrees, 0, 359.999999999)
      ) throw Error('Unsafe bearing guide')
      guideIndexes.add(props.detection_index as number)
    } else throw Error('Unexpected geographic feature')
  }
  if (cameraCount !== (cameraCoordinate ? 1 : 0)) throw Error('Camera marker missing or duplicated')

  if (!Array.isArray(data.observations) || data.observations.length > 512) throw Error('Invalid observations')
  const observed = new Set<number>()
  for (const obs of data.observations) {
    if (!isRecord(obs) || typeof obs.detection_index !== 'number' || !Number.isInteger(obs.detection_index) ||
        (obs.detection_index as number) < 0 || observed.has(obs.detection_index as number) ||
        !['drone', 'bird', 'unknown'].includes(String(obs.object_class)) ||
        !finite(obs.confidence, 0, 1) ||
        (obs.azimuth_degrees !== null && !finite(obs.azimuth_degrees, 0, 359.999999999)) ||
        !finite(obs.elevation_degrees, -90, 90) ||
        obs.north_reference !== request.north_reference ||
        obs.target_location !== null || obs.target_range_m !== null ||
        obs.target_location_observable !== false
    ) throw Error('Unsafe observation')
    observed.add(obs.detection_index as number)
  }
  for (const idx of guideIndexes) if (!observed.has(idx)) throw Error('Guide with no observation')
  return data as unknown as LiveMapScene
}
