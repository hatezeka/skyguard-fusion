// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { LiveStatus } from './api/liveCamera'
import LiveTrackHistoryPanel from './LiveTrackHistoryPanel'
afterEach(() => cleanup())
const sid = 'b'.repeat(32)
const status = (patch: Record<string, unknown> = {}): LiveStatus => ({
  state: 'running', frame_index: 12, camera_id: null, session_id: sid,
  tracks: [
    {track_id:1, class:'drone', state:'confirmed', confidence:.92, frame_index:12},
    {track_id:2, class:'bird', state:'confirmed', confidence:.91, frame_index:12},
  ],
  trajectories: [
    {track_id:1, object_class:'drone', points:[{frame_index:10,x_px:11,y_px:10,timestamp:"2026-11-20T10:00:00Z",velocity_x_px_s:0,velocity_y_px_s:0,speed_px_s:0,confidence:.92},{frame_index:11,x_px:20,y_px:30,timestamp:"2026-11-20T10:00:01Z",velocity_x_px_s:9,velocity_y_px_s:20,speed_px_s:21.93,confidence:.92}]},
    {track_id:2, object_class:'bird', points:[{frame_index:10,x_px:45,y_px:10,timestamp:"2026-11-20T10:00:00Z",velocity_x_px_s:0,velocity_y_px_s:0,speed_px_s:0,confidence:.91},{frame_index:11,x_px:30,y_px:30,timestamp:"2026-11-20T10:00:01Z",velocity_x_px_s:-15,velocity_y_px_s:20,speed_px_s:25,confidence:.91}]},
  ],
  ambiguous_tracks: [], detections: [], error:null, processing_seconds:null, ...patch,
} as LiveStatus)

describe('Phase 40.3 live camera pixel trajectories', () => {
  it('does not show tracks when disconnected', () => {
    render(<LiveTrackHistoryPanel liveStatus={status()} connected={false} />)
    expect(screen.getByText(/API e shkëputur/)).toBeTruthy()
    expect(screen.queryByText(/Dronë të konfirmuar/)).toBeNull()
  })
  it('shows two distinct track ids and no GPS claims', () => {
    render(<LiveTrackHistoryPanel liveStatus={status()} connected />)
    expect(screen.getByText(/Dronë të konfirmuar \(aktivë\): 1/)).toBeTruthy()
    expect(screen.getAllByText('ID 1').length).toBeGreaterThan(0)
    expect(screen.getAllByText('ID 2').length).toBeGreaterThan(0)
    expect(screen.getByText(/jo trajektore GPS/)).toBeTruthy()
  })
  it('focuses a track without joining crossing paths', () => {
    render(<LiveTrackHistoryPanel liveStatus={status()} connected />)
    fireEvent.click(screen.getByRole('button', {name:'ID 2'}))
    expect(screen.getByRole('button', {name:'ID 2'}).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByRole('img', {name:/Trajektore të objekteve/}).querySelectorAll('polyline').length).toBe(1)
  })
  it('clears selection across session identity changes', () => {
    const { rerender } = render(<LiveTrackHistoryPanel liveStatus={status()} connected />)
    fireEvent.click(screen.getByRole('button', {name:'ID 2'}))
    rerender(<LiveTrackHistoryPanel liveStatus={status({session_id:'c'.repeat(32)})} connected />)
    expect(screen.getByRole('button', {name:'Të gjitha'}).getAttribute('aria-pressed')).toBe('true')
  })
  it('shows ambiguous associations separately', () => {
    render(<LiveTrackHistoryPanel liveStatus={status({ambiguous_tracks:[{state:'ambiguous',frame_index:12,candidate_track_ids:[1,2]}]})} connected />)
    expect(screen.getAllByText(/asociime të paqarta/).length).toBeGreaterThan(0)
  })
  it('rejects malformed data without rendering GPS or trajectories', () => {
    render(<LiveTrackHistoryPanel liveStatus={status({session_id:'invalid'})} connected />)
    expect(screen.getByRole('alert')).toBeTruthy()
    expect(screen.queryByRole('img', {name:/Trajektore të objekteve/})).toBeNull()
  })
})
