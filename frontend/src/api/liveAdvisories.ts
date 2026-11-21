/** Evidence-only alerts: no inferred coordinates, physical identity or geofence. */
export interface AdvisoryEvent {
  event_id: string
  track_id: number
  frame_index: number
  confidence: number
  session_id: string
  camera_id: string
  event_type: 'operator_review_advisory'
  severity: 'attention'
  object_class_model_label: 'drone'
  reason_codes: string[]
  ground_truth_verified: false
  facility_breach_verified: false
  target_gps: null
  target_range_m: null
}
export interface AdvisoryBatch {
  camera_id: string
  session_id: string
  frame_index: number
  alerts: AdvisoryEvent[]
  sampled_frames_only: true
  alert_archive_persistent: false
}
export interface AlertHistory {
  key: string
  frame: number
  events: AdvisoryEvent[]
  error: string | null
}
const SESSION = /^[a-f0-9]{32}$/
const CAMERA = /^[a-zA-Z0-9_-]{1,128}$/
const HASH = /^[a-f0-9]{64}$/
const REASON = /^[A-Z][A-Z0-9_]{1,80}$/
const obj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const index = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0
const confidence = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1
export function alertScope(camera: string, session: string): string {
  if (!CAMERA.test(camera) || !SESSION.test(session)) throw new Error('Invalid session identity')
  return `${camera}:${session}`
}
export function parseAdvisory(value: unknown, camera: string, session: string): AdvisoryBatch {
  alertScope(camera, session)
  if (!obj(value) || value.camera_id !== camera || value.session_id !== session ||
      !index(value.frame_index) || value.evaluation_basis !== 'model_and_tracker_evidence_only' ||
      value.coordinate_system !== 'image_pixels' || value.sampling_mode !== 'on_demand_latest_snapshot' ||
      value.sampled_frames_only !== true || value.alert_archive_persistent !== false ||
      value.ground_truth_verified !== false || value.target_gps_available !== false ||
      value.facility_distance_available !== false || value.geofence_breach_verified !== false ||
      value.automated_response_allowed !== false ||
      !Array.isArray(value.alerts) || value.alerts.length > 512) {
    throw new Error('Unsafe or stale advisory response')
  }
  const seen = new Set<string>()
  const alerts: AdvisoryEvent[] = []
  for (const event of value.alerts) {
    if (!obj(event) || typeof event.event_id !== 'string' || !HASH.test(event.event_id) ||
        seen.has(event.event_id) || !index(event.track_id) || !index(event.frame_index) ||
        event.frame_index > value.frame_index || event.camera_id !== camera || event.session_id !== session ||
        event.event_type !== 'operator_review_advisory' || event.severity !== 'attention' ||
        event.object_class_model_label !== 'drone' || !confidence(event.confidence) ||
        !Array.isArray(event.reason_codes) || event.reason_codes.length > 12 ||
        event.reason_codes.some(r => typeof r !== 'string' || !REASON.test(r)) ||
        event.ground_truth_verified !== false || event.facility_breach_verified !== false ||
        event.target_gps !== null || event.target_range_m !== null) {
      throw new Error('Unsafe advisory event')
    }
    seen.add(event.event_id)
    alerts.push({
      event_id: event.event_id, track_id: event.track_id, frame_index: event.frame_index,
      confidence: event.confidence, session_id: session, camera_id: camera,
      event_type: 'operator_review_advisory', severity: 'attention', object_class_model_label: 'drone',
      reason_codes: event.reason_codes.slice(), ground_truth_verified: false,
      facility_breach_verified: false, target_gps: null, target_range_m: null,
    })
  }
  return {
    camera_id: camera, session_id: session, frame_index: value.frame_index,
    alerts, sampled_frames_only: true, alert_archive_persistent: false,
  }
}
/** Bounded process-local UI history, session-scoped and deduplicated. */
export function mergeAdvisories(previous: AlertHistory | null, response: AdvisoryBatch): AlertHistory {
  const key = alertScope(response.camera_id, response.session_id)
  const prior = previous?.key === key ? previous : null
  if (prior && response.frame_index < prior.frame) return prior
  const ids = new Set(prior?.events.map(e => e.event_id) ?? [])
  const novel = response.alerts.filter(e => !ids.has(e.event_id))
  return { key, frame: response.frame_index, events: [...novel, ...(prior?.events ?? [])].slice(0, 80), error: null }
}
