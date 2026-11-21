import type { LiveMapScene, NorthReference } from './liveMapScene'
export type LiveCameraSource =
  | {
      source_type: 'usb'
      source_location: number
    }
  | {
      source_type: 'rtsp'
      source_location: string
    }
export const LIVE_API = '/api/v1/live'

export interface LiveBoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface LiveDetection {
  class: string
  confidence: number
  bbox: LiveBoundingBox
}

export interface LiveTrack {
  track_id: number
  class: string
  confidence: number
  state: string
  frame_index?: number
}

export interface LiveAmbiguousTrack {
  class: string
  confidence: number
  state: 'ambiguous'
  frame_index: number
  candidate_track_ids: number[]
}

export interface LiveTrajectoryPoint {
  frame_index: number
  timestamp: string
  x_px: number
  y_px: number
  velocity_x_px_s: number | null
  velocity_y_px_s: number | null
  speed_px_s: number | null
  confidence: number
}

export interface LiveTrajectory {
  track_id: number
  object_class: string
  points: LiveTrajectoryPoint[]
}

export interface LiveStatus {
  camera_id?: string | null
  session_id?: string | null

  state: string
  frame_index: number | null
  detections: LiveDetection[]
  tracks: LiveTrack[]
  ambiguous_tracks?: LiveAmbiguousTrack[]
  trajectories?: LiveTrajectory[]
  processing_seconds: number | null
  error: string | null
}


export interface LiveSynchronizedFrame {
  state: 'running'
  frame_index: number
  image_mime: 'image/jpeg'
  image_base64: string
  tracks: LiveTrack[]
  trajectories: LiveTrajectory[]
}

export interface LiveDetections {
  state?: string
  frame_index?: number | null
  detections: LiveDetection[]
  tracks?: LiveTrack[]
  ambiguous_tracks?: LiveAmbiguousTrack[]
  trajectories?: LiveTrajectory[]
  processing_seconds?: number | null
  error?: string | null
}

async function request<T>(
  endpoint: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${LIVE_API}${endpoint}`, {
    cache: 'no-store',
    ...options,
  })

  if (!response.ok) {
    const body = await response.text()
    throw new Error(
      `HTTP ${response.status}: ${body || response.statusText}`,
    )
  }

  return response.json() as Promise<T>
}

export const liveCameraApi = {
  status: () => request<LiveStatus>('/status'),

  frame: () => request<LiveSynchronizedFrame>('/frame'),

  detections: () =>
    request<LiveDetections>('/detections'),

  start: (
    cameraIndex = 0,
    confidence = 0.35,
    source?: LiveCameraSource,
    fusionEnabled = false,
  ) =>
    request<LiveStatus>('/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        camera_index: cameraIndex,
        confidence,
        fusion_enabled: fusionEnabled,
        ...(source
          ? {
              source_type: source.source_type,
              source_location: source.source_location,
            }
          : {}),
      }),
    }),

  stop: () =>
    request<LiveStatus>('/stop', {
      method: 'POST',
    }),

  // Phase 40.2: GET selects a synchronized frame in the backend.
  mapSceneLatest: (northReference: NorthReference) =>
    request<LiveMapScene>(`/map/scene/latest?north_reference=${encodeURIComponent(northReference)}`),

  // Phase 41.3: uses the existing authenticated request helper.
  evaluateLatestAdvisories: () =>
    request<unknown>('/alerts/evaluate/latest', { method: 'POST' }),

  streamUrl: `${LIVE_API}/stream`,
}
