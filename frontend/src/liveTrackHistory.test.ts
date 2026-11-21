import { describe, expect, it } from 'vitest'
import type { LiveStatus } from './api/liveCamera'
import { parsePixelTrackHistory } from './api/liveTrackHistory'
const SID = 'a'.repeat(32)
const status = (override: Record<string, unknown> = {}): LiveStatus => ({
  state: 'running', frame_index: 42, camera_id: null, session_id: SID,
  tracks: [{ track_id: 3, class: 'drone', confidence: 0.92, state: 'confirmed', frame_index: 42 }],
  trajectories: [{ track_id: 3, object_class: 'drone', points: [
    { frame_index: 40, x_px: 10, y_px: 100, timestamp: '2026-11-20T10:00:00Z', velocity_x_px_s: 0, velocity_y_px_s: 0, speed_px_s: 0, confidence: .92 },
    { frame_index: 41, x_px: 20, y_px: 95, timestamp: '2026-11-20T10:00:01Z', velocity_x_px_s: 10, velocity_y_px_s: -5, speed_px_s: 11.18, confidence: .92 },
  ] }], ambiguous_tracks: [], detections: [], error: null, processing_seconds: null,
  ...override,
} as LiveStatus)

describe('Phase 40.3 pixel track identity and provenance', () => {
  it('returns null when session is not running', () => {
    expect(parsePixelTrackHistory(status({ state: 'stopped' }))).toBeNull()
  })
  it('works without a registered camera (built-in laptop)', () => {
    const result = parsePixelTrackHistory(status())!
    expect(result.camera_id).toBeNull()
    expect(result.tracks[0].track_id).toBe(3)
    expect(result.current_drone_count).toBe(1)
    expect(result.coordinate_system).toBe('image_pixels')
    expect(result.target_gps_available).toBe(false)
    expect(result.target_range_available).toBe(false)
  })
  it('keeps multiple IDs separate when trajectories cross', () => {
    const result = parsePixelTrackHistory(status({
      tracks: [
        { track_id: 3, class: 'drone', confidence: .9, state: 'confirmed', frame_index: 42 },
        { track_id: 4, class: 'drone', confidence: .8, state: 'confirmed', frame_index: 42 },
      ],
      trajectories: [
        { track_id: 3, points: [{frame_index: 40,x_px:10,y_px:10},{frame_index:41,x_px:30,y_px:30}] },
        { track_id: 4, points: [{frame_index:40,x_px:30,y_px:10},{frame_index:41,x_px:10,y_px:30}] },
      ],
    }))!
    expect(result.tracks.map(t => t.track_id)).toEqual([3,4])
    expect(result.tracks[0].points[1].x_px).toBe(30)
    expect(result.tracks[1].points[1].x_px).toBe(10)
  })
  it('never turns ambiguous candidates into a merged track', () => {
    const result = parsePixelTrackHistory(status({
      ambiguous_tracks: [{ state:'ambiguous', frame_index:42, candidate_track_ids:[3,7] }],
    }))!
    expect(result.ambiguous_count).toBe(1)
    expect(result.ambiguous_candidate_ids).toEqual([3,7])
    expect(result.tracks.length).toBe(1)
  })
  it('historical-only ID has no inferred drone class', () => {
    const result = parsePixelTrackHistory(status({tracks: [], trajectories:[{track_id:9,points:[{frame_index:39,x_px:4,y_px:5}]}]}))!
    expect(result.tracks[0].object_class).toBe('unknown')
    expect(result.tracks[0].active).toBe(false)
    expect(result.current_drone_count).toBe(0)
  })
  it('rejects duplicate track ids', () => {
    const track = { track_id: 3, class: 'drone', confidence: 0.9, state: 'confirmed', frame_index: 42 }
    expect(() => parsePixelTrackHistory(status({ tracks:[track,track] }))).toThrow()
  })
  it('rejects duplicated trajectory ownership', () => {
    const trail = {track_id:3,points:[]}
    expect(() => parsePixelTrackHistory(status({ trajectories:[trail,trail] }))).toThrow()
  })
  it('rejects future, unordered or duplicated frame points', () => {
    for (const frames of [[41, 40], [41, 41], [41, 43]]) {
      expect(() => parsePixelTrackHistory(status({ trajectories: [{track_id:3,points:frames.map(n => ({frame_index:n,x_px:2,y_px:3}))}] }))).toThrow()
    }
  })
  it('rejects a broken session and spoofed camera id', () => {
    expect(() => parsePixelTrackHistory(status({session_id:'fake'}))).toThrow()
    expect(() => parsePixelTrackHistory(status({camera_id:'../../etc'}))).toThrow()
  })
  it('rejects nonfinite or negative coordinates', () => {
    for (const x of [NaN, Infinity, -1]) {
      expect(() => parsePixelTrackHistory(status({trajectories:[{track_id:3,points:[{frame_index:41,x_px:x,y_px:4}]}]}))).toThrow()
    }
  })
  it('rejects invalid confidence and future track frame', () => {
    for (const change of [{confidence: 2}, {frame_index:43}]) {
      expect(() => parsePixelTrackHistory(status({tracks:[{track_id:3,class:'drone',confidence:.9,state:'confirmed',frame_index:42,...change}]}))).toThrow()
    }
  })
  it('keeps classes separate from actual confirmed drone count', () => {
    const result = parsePixelTrackHistory(status({tracks: [
      {track_id:1,class:'bird',confidence:.9,state:'confirmed',frame_index:42},
      {track_id:2,class:'drone',confidence:.9,state:'tentative',frame_index:42},
    ], trajectories:[]}))!
    expect(result.current_drone_count).toBe(0)
    expect(result.tracks.length).toBe(2)
  })
  it('refuses oversized arbitrary arrays', () => {
    expect(() => parsePixelTrackHistory(status({tracks:Array(513).fill({})}))).toThrow()
  })
})
