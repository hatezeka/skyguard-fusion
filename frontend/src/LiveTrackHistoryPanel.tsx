import { useState } from 'react'
import type { LiveStatus } from './api/liveCamera'
import { parsePixelTrackHistory, type PixelTrack, type PixelTrackHistory } from './api/liveTrackHistory'
import './LiveTrackHistoryPanel.css'

interface Props {
  liveStatus: LiveStatus
  connected: boolean
}
const MAX_DISPLAY = 64
const WIDTH = 760
const HEIGHT = 300
const PAD = 24

function validView(status: LiveStatus): { view: PixelTrackHistory | null; error: string | null } {
  try { return { view: parsePixelTrackHistory(status), error: null } }
  catch { return { view: null, error: 'Historiku i gjurmimit u refuzua: të dhëna ose identitete të pavlefshme.' } }
}

export default function LiveTrackHistoryPanel({ liveStatus, connected }: Props) {
  const parsed = validView(liveStatus)
  const [selection, setSelection] = useState<{ sessionId: string; trackId: number } | null>(null)
  const view = connected ? parsed.view : null
  const missingSession = liveStatus.state === 'running' &&
    (liveStatus.session_id === null || liveStatus.session_id === undefined)
  const requestedId = selection && view && selection.sessionId === view.session_id ? selection.trackId : null
  const selectedId = requestedId !== null && view?.tracks.some(t => t.track_id === requestedId) ? requestedId : null
  const tracks = view?.tracks.slice(0, MAX_DISPLAY) ?? []
  const visible = selectedId === null ? tracks : tracks.filter(t => t.track_id === selectedId)
  const pts = tracks.flatMap(track => track.points)
  let right = 640
  let bottom = 360
  for (const p of pts) { right = Math.max(right, p.x_px); bottom = Math.max(bottom, p.y_px) }
  const sx = (WIDTH - 2 * PAD) / Math.max(1, right)
  const sy = (HEIGHT - 2 * PAD) / Math.max(1, bottom)
  const transform = (x: number, y: number) => `${(PAD + x * sx).toFixed(2)},${(PAD + y * sy).toFixed(2)}`
  const geometry = { transform, right, bottom }

  return (
    <section className="skyguard-track-panel" aria-label="Historiku live i gjurmëve të kamerës">
      <div className="skyguard-track-header">
        <div>
          <h2>LIVE TRACK HISTORY</h2>
          <p>Gjurmë në pikselë të kamerës, sipas ID-ve të tracker-it — jo trajektore GPS.</p>
        </div>
      </div>
      {!connected && <p role="status">API e shkëputur; historiku live nuk është aktiv.</p>}
      {connected && liveStatus.state !== 'running' && <p role="status">Nis kamerën për të parë trajektoret.</p>}
      {connected && missingSession && <p role="status">Historiku kërkon session_id nga backend-i.</p>}
      {connected && liveStatus.state === 'running' && parsed.error && !missingSession && <p role="alert">{parsed.error}</p>}
      {view && <>
        <div className="skyguard-track-stats">
          <span>Sesioni: {view.session_id.slice(0, 8)}…</span>
          <span>Frame: {view.frame_index}</span>
          <span>Gjurmë të raportuara: {view.tracks.length}</span>
          <span>Dronë të konfirmuar (aktivë): {view.current_drone_count}</span>
          <span>Asociime të paqarta: {view.ambiguous_count}</span>
        </div>
        <div className="skyguard-track-actions">
          <button type="button" aria-pressed={selectedId === null} onClick={() => setSelection(null)}>Të gjitha</button>
          {tracks.map(track => (
            <button type="button" key={track.track_id} aria-pressed={selectedId === track.track_id}
              onClick={() => setSelection({ sessionId: view.session_id, trackId: track.track_id })}>
              ID {track.track_id}
            </button>
          ))}
        </div>
        {view.tracks.length > MAX_DISPLAY && <p role="note">Vizualizohen {MAX_DISPLAY} nga {view.tracks.length} gjurmë; të dhënat e tjera mbeten në status.</p>}
        <div className="skyguard-track-canvas">
          <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Trajektore të objekteve në koordinatat e imazhit, jo pozicione gjeografike">
            <rect x="0" y="0" width={WIDTH} height={HEIGHT} className="skyguard-track-bg" />
            {Array.from({ length: 9 }, (_, i) => <line key={`v${i}`} x1={PAD + i * (WIDTH - 2 * PAD) / 8} y1={PAD} x2={PAD + i * (WIDTH - 2 * PAD) / 8} y2={HEIGHT - PAD} className="skyguard-track-grid" />)}
            {Array.from({ length: 5 }, (_, i) => <line key={`h${i}`} x1={PAD} y1={PAD + i * (HEIGHT - 2 * PAD) / 4} x2={WIDTH - PAD} y2={PAD + i * (HEIGHT - 2 * PAD) / 4} className="skyguard-track-grid" />)}
            {visible.map((track: PixelTrack) => {
              const points = track.points
              const last = points.length ? points[points.length - 1] : undefined
              return <g key={track.track_id}>
                {points.length > 1 && <polyline style={{ stroke: `hsl(${(track.track_id * 137.508) % 360}, 75%, 67%)` }} className={track.active ? 'skyguard-track-line' : 'skyguard-track-line skyguard-track-inactive'}
                  points={points.map(p => geometry.transform(p.x_px, p.y_px)).join(' ')} />}
                {last && <circle cx={Number(geometry.transform(last.x_px, last.y_px).split(',')[0])}
                  cy={Number(geometry.transform(last.x_px, last.y_px).split(',')[1])} r={5} className="skyguard-track-dot" />}
                {last && <text x={Number(geometry.transform(last.x_px, last.y_px).split(',')[0]) + 9}
                  y={Number(geometry.transform(last.x_px, last.y_px).split(',')[1]) - 6} className="skyguard-track-id">#{track.track_id}</text>}
              </g>
            })}
            <text x={12} y={HEIGHT - 6} className="skyguard-track-caption">Koordinata pikselësh • shtrirje x: 0–{Math.ceil(geometry.right)}, y: 0–{Math.ceil(geometry.bottom)}</text>
          </svg>
        </div>
        <div className="skyguard-track-list" aria-label="Identitetet e gjurmëve">
          {visible.map(track => <div key={track.track_id} className="skyguard-track-record">
            <strong>ID {track.track_id}</strong>
            <span>{track.object_class}</span>
            <span>{track.active ? track.state : 'historik; jo aktiv'}</span>
            <span>{track.points.length} pika</span>
            {track.confidence !== null && <span>{(track.confidence * 100).toFixed(0)}% besueshmëri modeli</span>}
          </div>)}
        </div>
        {view.ambiguous_count > 0 && <p role="note" className="skyguard-track-ambiguous">
          {view.ambiguous_count} asociime të paqarta. Kandidatë ID: {view.ambiguous_candidate_ids.length ? view.ambiguous_candidate_ids.join(', ') : 'nuk dihen'}. Nuk bashkohen automatikisht.
        </p>}
        <p role="note" className="skyguard-track-disclaimer">
          Historiku vjen nga tracker-i i sesionit aktual; ID-të janë ID tracker-i, jo identitet fizik i verifikuar.
          Pozicioni GPS dhe distanca e dronit nuk maten nga kjo pamje. Kryqëzimet ose humbja e një ID-je kërkojnë validim të veçantë.
        </p>
      </>}
    </section>
  )
}
