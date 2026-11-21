/**
 * SkyGuard evidence API client. Keep bearer tokens in process memory only; NEVER
 * put a database service key in VITE_ variables, localStorage or checked-in files.
 * For production use an authenticated BFF / reverse proxy and user sessions.
 */
export type SkyGuardClass = 'DRONE' | 'BIRD' | 'UNKNOWN'
export type ValidationLevel = 'SOFTWARE_ONLY' | 'PHYSICAL_UNVERIFIED' | 'PHYSICAL_VERIFIED'
export interface CameraRecord {
  id: string
  name: string
  source_kind: 'USB' | 'RTSP' | 'FILE' | 'SIMULATED' | 'OTHER'
  latitude: number | null
  longitude: number | null
  calibration_ref: string | null
  created_at: string
}
export interface SessionRecord {
  id: string
  camera_id: string
  source_kind: 'LIVE' | 'VIDEO' | 'IMAGE' | 'SIMULATION'
  validation_level: ValidationLevel
  started_at: string
  ended_at: string | null
}
export interface DetectionRecord {
  id: string
  frame_id: string
  session_id: string
  frame_index: number
  captured_at: string
  classification: SkyGuardClass
  confidence: number | null
  x1: number; y1: number; x2: number; y2: number
  track_id: string | null
}
export interface TrackRecord {
  id: string
  session_id: string
  external_track_id: string
  classification: 'DRONE' | 'BIRD'
  first_seen: string
  last_seen: string
}
export interface GeospatialRecord {
  id: string
  detection_id: string
  session_id: string
  classification: 'DRONE' | 'BIRD'
  method: string
  evidence_ref: string
  calibration_ref: string
  latitude: number | null
  longitude: number | null
  range_m: number | null
  uncertainty_m: number
}
export interface AlertRecord {
  id: string
  session_id: string
  detection_id: string | null
  severity: 'INFO' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  reason: string
  source: string
  status: 'OPEN' | 'ACKNOWLEDGED'
  acknowledged_by: string | null
  created_at: string
}
export interface SummaryRecord {
  session_id: string
  frames: number
  tracks: number
  alerts: number
  detections: Partial<Record<SkyGuardClass, number>>
  note: string
}
export interface PerformanceRecord {
  id: string
  session_id: string
  fps: number | null
  latency_ms: number | null
  method: string
  stage: string
  measured_at: string
}

/** Restrict URL choices to actual API deployment; defaults to same-origin. */
export class SkyGuardPersistenceClient {
  private readonly getBearerToken: () => string | null
  private readonly baseUrl: string

  constructor(getBearerToken: () => string | null, baseUrl = '') {
    this.getBearerToken = getBearerToken
    this.baseUrl = baseUrl
  }

  private async get<T>(path: string, query?: Record<string, string | number | undefined>): Promise<T> {
    const token = this.getBearerToken()
    const deployed = import.meta.env.VITE_SKYGUARD_DEPLOYED === 'true'
    if (!deployed && !token) throw new Error('Authenticated persistence session required')
    const qs = new URLSearchParams()
    Object.entries(query ?? {}).forEach(([key, value]) => {
      if (value !== undefined) qs.set(key, String(value))
    })
    const response = await fetch(`${this.baseUrl}/api/persistence${path}${qs.size ? `?${qs}` : ''}`, {
      headers: deployed ? {} : { Authorization: `Bearer ${token}` },
      credentials: 'same-origin',
      cache: 'no-store',
    })
    if (!response.ok) throw new Error(`SkyGuard persistence HTTP ${response.status}`)
    return (await response.json()) as T
  }

  cameras() { return this.get<CameraRecord[]>('/cameras') }
  sessions() { return this.get<SessionRecord[]>('/sessions') }
  summary(sessionId: string) { return this.get<SummaryRecord>(`/sessions/${encodeURIComponent(sessionId)}/summary`) }
  detections(sessionId?: string, classification?: SkyGuardClass, limit = 100) {
    return this.get<DetectionRecord[]>('/detections', { session_id: sessionId, classification, limit })
  }
  tracks(sessionId?: string) { return this.get<TrackRecord[]>('/tracks', { session_id: sessionId }) }
  history(trackId: string) {
    return this.get<Array<{ observation: { id: string; observed_at: string; frame_index: number }; detection: DetectionRecord }>>(
      `/tracks/${encodeURIComponent(trackId)}/history`)
  }
  geospatial(sessionId?: string) { return this.get<GeospatialRecord[]>('/geospatial', { session_id: sessionId }) }
  alerts(sessionId?: string) { return this.get<AlertRecord[]>('/alerts', { session_id: sessionId }) }
  performance(sessionId: string) { return this.get<PerformanceRecord[]>('/performance', { session_id: sessionId }) }
}
