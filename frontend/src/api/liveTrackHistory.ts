import type { LiveStatus } from './liveCamera'

/** Camera-image tracks only. Never geographical paths or measured ranges. */
export interface PixelTrailPoint {
  frame_index: number
  x_px: number
  y_px: number
}
export interface PixelTrack {
  track_id: number
  object_class: string
  confidence: number | null
  state: string
  active: boolean
  points: PixelTrailPoint[]
}
export interface PixelTrackHistory {
  session_id: string
  camera_id: string | null
  frame_index: number
  coordinate_system: 'image_pixels'
  target_gps_available: false
  target_range_available: false
  tracks: PixelTrack[]
  ambiguous_count: number
  ambiguous_candidate_ids: number[]
  current_drone_count: number
}
const SESSION = /^[0-9a-f]{32}$/
const CAM = /^[a-zA-Z0-9_-]{1,128}$/
const CLASS = /^[a-zA-Z][a-zA-Z0-9_-]{0,30}$/
const STATE = /^[a-zA-Z][a-zA-Z0-9_-]{0,30}$/
const MAX_TRACKS = 512
const MAX_POINTS = 512
function record(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}
function integer(v: unknown, min: number, max: number): v is number {
  return typeof v === 'number' && Number.isSafeInteger(v) && v >= min && v <= max
}
function pixel(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1000000
}
function confidence(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1
}

/**
 * This parser never joins detections based on geometry or proximity. It only
 * displays explicit tracker IDs supplied by the same atomic status snapshot.
 * Ambiguous candidates remain separate; lost IDs cannot be guessed.
 */
export function parsePixelTrackHistory(status: LiveStatus): PixelTrackHistory | null {
  if (status.state !== 'running') return null
  if (typeof status.session_id !== 'string' || !SESSION.test(status.session_id) ||
      !integer(status.frame_index, 0, Number.MAX_SAFE_INTEGER) ||
      (status.camera_id !== null && status.camera_id !== undefined &&
        (typeof status.camera_id !== 'string' || !CAM.test(status.camera_id)))) {
    throw new Error('Invalid session or camera identity')
  }
  const frame = status.frame_index as number
  const rawTracks: unknown = status.tracks
  const rawPaths: unknown = status.trajectories ?? []
  const rawAmbiguous: unknown = status.ambiguous_tracks ?? []
  if (!Array.isArray(rawTracks) || rawTracks.length > MAX_TRACKS ||
      !Array.isArray(rawPaths) || rawPaths.length > MAX_TRACKS ||
      !Array.isArray(rawAmbiguous) || rawAmbiguous.length > MAX_TRACKS) {
    throw new Error('Invalid tracking payload')
  }
  const active = new Map<number, PixelTrack>()
  for (const raw of rawTracks) {
    if (!record(raw) || !integer(raw.track_id, 0, Number.MAX_SAFE_INTEGER) ||
        active.has(raw.track_id) || typeof raw.class !== 'string' || !CLASS.test(raw.class) ||
        !confidence(raw.confidence) || typeof raw.state !== 'string' || !STATE.test(raw.state) ||
        !integer(raw.frame_index, 0, frame)) {
      throw new Error('Invalid or duplicated tracker identity')
    }
    active.set(raw.track_id, {
      track_id: raw.track_id, object_class: raw.class,
      confidence: raw.confidence, state: raw.state, active: raw.frame_index === frame, points: [],
    })
  }
  const seenPathIds = new Set<number>()
  for (const raw of rawPaths) {
    if (!record(raw) || !integer(raw.track_id, 0, Number.MAX_SAFE_INTEGER) ||
        seenPathIds.has(raw.track_id) || !Array.isArray(raw.points) ||
        raw.points.length > MAX_POINTS) {
      throw new Error('Invalid or duplicated trajectory identity')
    }
    seenPathIds.add(raw.track_id)
    const points: PixelTrailPoint[] = []
    let previous = -1
    for (const p of raw.points) {
      if (!record(p) || !integer(p.frame_index, 0, frame) || p.frame_index <= previous ||
          !pixel(p.x_px) || !pixel(p.y_px)) {
        throw new Error('Invalid, unordered or future trajectory point')
      }
      previous = p.frame_index
      points.push({ frame_index: p.frame_index, x_px: p.x_px, y_px: p.y_px })
    }
    const existing = active.get(raw.track_id)
    if (existing) existing.points = points
    else active.set(raw.track_id, {
      track_id: raw.track_id, object_class: 'unknown', confidence: null,
      state: 'historical', active: false, points,
    })
  }
  const candidates = new Set<number>()
  for (const ambiguous of rawAmbiguous) {
    if (!record(ambiguous) || ambiguous.state !== 'ambiguous' ||
        !integer(ambiguous.frame_index, 0, frame) ||
        !Array.isArray(ambiguous.candidate_track_ids) ||
        ambiguous.candidate_track_ids.length > MAX_TRACKS) {
      throw new Error('Invalid ambiguous tracking record')
    }
    for (const id of ambiguous.candidate_track_ids) {
      if (!integer(id, 0, Number.MAX_SAFE_INTEGER)) throw new Error('Invalid ambiguous candidate ID')
      candidates.add(id)
    }
  }
  return {
    session_id: status.session_id,
    camera_id: status.camera_id ?? null,
    frame_index: frame,
    coordinate_system: 'image_pixels',
    target_gps_available: false,
    target_range_available: false,
    tracks: [...active.values()].sort((a, b) => a.track_id - b.track_id),
    ambiguous_count: rawAmbiguous.length,
    ambiguous_candidate_ids: [...candidates].sort((a, b) => a - b),
    current_drone_count: [...active.values()].filter(t => t.active && t.object_class === 'drone' && t.state === 'confirmed').length,
  }
}
