// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import LiveMapScenePanel from './LiveMapScenePanel'
import { liveCameraApi, type LiveStatus } from './api/liveCamera'

vi.mock('./api/liveCamera', () => ({ liveCameraApi: { mapSceneLatest: vi.fn() } }))
const ID = '0123456789abcdef0123456789abcdef'
const newStatus = (overrides: Record<string, unknown> = {}): LiveStatus => ({
  state: 'running', frame_index: 8, session_id: ID, camera_id: 'cam-1',
  detections: [], tracks: [], processing_seconds: null, error: null,
  ...overrides,
} as LiveStatus)
const response = () => ({
  scene_type: 'live_camera_bearing_map', camera_id: 'cam-1', session_id: ID,
  frame_index: 8, north_reference: 'magnetic_north', calibration_snapshot_sha256: 'a'.repeat(64),
  session_identity_verified: true, calibration_bound_to_session_verified: true,
  calibration_bound_to_frame_verified: true, physical_calibration_verified: false,
  camera_location_bound_to_session_verified: false, camera_location_survey_verified: false,
  camera_location_source: 'unavailable', camera_location: null,
  target_gps_available: false, target_range_available: false,
  bearing_guides_are_display_only: true, magnetic_declination_applied: false,
  observations: [], geojson: { type: 'FeatureCollection', features: [] },
})

beforeEach(() => vi.clearAllMocks())
afterEach(() => cleanup())
describe('Phase 40.2 live scene panel', () => {
  it('does not query when backend is disconnected', () => {
    render(<LiveMapScenePanel liveStatus={newStatus()} connected={false} />)
    expect(screen.getByText(/API e shkëputur/)).toBeTruthy()
    expect(liveCameraApi.mapSceneLatest).not.toHaveBeenCalled()
  })
  it('refuses to infer identity when the camera is unregistered', () => {
    render(<LiveMapScenePanel liveStatus={newStatus({ camera_id: null })} connected />)
    expect(screen.getByText(/duhet të jetë e regjistruar/)).toBeTruthy()
    expect(liveCameraApi.mapSceneLatest).not.toHaveBeenCalled()
  })
  it('requests magnetic north without geographic guides until user confirms a reference', async () => {
    vi.mocked(liveCameraApi.mapSceneLatest).mockResolvedValue(response() as never)
    render(<LiveMapScenePanel liveStatus={newStatus()} connected />)
    await waitFor(() => expect(screen.getByText(/Nuk ka koordinata të deklaruara/)).toBeTruthy())
    expect(liveCameraApi.mapSceneLatest).toHaveBeenCalledWith('magnetic_north')
  })
  it('does not display spoofed target GPS', async () => {
    vi.mocked(liveCameraApi.mapSceneLatest).mockResolvedValue({ ...response(), target_gps_available: true } as never)
    render(<LiveMapScenePanel liveStatus={newStatus()} connected />)
    await waitFor(() => expect(screen.getByRole('alert')).toBeTruthy())
    expect(screen.queryByText(/Detektime: 0/)).toBeNull()
  })
  it('does not render the scene of a previous session', async () => {
    vi.mocked(liveCameraApi.mapSceneLatest).mockResolvedValue(response() as never)
    const { rerender } = render(<LiveMapScenePanel liveStatus={newStatus()} connected />)
    await waitFor(() => expect(screen.getByText(/Nuk ka koordinata të deklaruara/)).toBeTruthy())
    rerender(<LiveMapScenePanel liveStatus={newStatus({ session_id: 'b'.repeat(32) })} connected />)
    expect(screen.queryByText(/Nuk ka koordinata të deklaruara/)).toBeNull()
  })
})
