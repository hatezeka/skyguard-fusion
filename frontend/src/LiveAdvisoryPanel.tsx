import { useEffect, useState } from 'react'
import { liveCameraApi, type LiveStatus } from './api/liveCamera'
import { alertScope, mergeAdvisories, parseAdvisory, type AlertHistory } from './api/liveAdvisories'
import './LiveAdvisoryPanel.css'

interface Props { liveStatus: LiveStatus; connected: boolean }
const SESSION = /^[a-f0-9]{32}$/
const CAMERA = /^[a-zA-Z0-9_-]{1,128}$/
export default function LiveAdvisoryPanel({ liveStatus, connected }: Props) {
  const [history, setHistory] = useState<AlertHistory | null>(null)
  const cam = liveStatus.camera_id
  const sid = liveStatus.session_id
  const frame = liveStatus.frame_index
  const ready = connected && liveStatus.state === 'running' &&
    typeof cam === 'string' && CAMERA.test(cam) && typeof sid === 'string' && SESSION.test(sid) &&
    typeof frame === 'number' && Number.isSafeInteger(frame) && frame >= 0
  const scope = ready ? alertScope(cam, sid) : null

  useEffect(() => {
    if (!scope || !cam || !sid) return
    let active = true
    let busy = false
    const poll = async () => {
      if (busy) return
      busy = true
      try {
        const raw = await liveCameraApi.evaluateLatestAdvisories()
        if (!active) return
        const result = parseAdvisory(raw, cam, sid)
        setHistory(previous => mergeAdvisories(previous, result))
      } catch {
        if (active) setHistory(previous => ({
          key: scope, frame: previous?.key === scope ? previous.frame : -1,
          events: previous?.key === scope ? previous.events : [],
          error: 'Sinjalizimet nuk u përditësuan. Kontrollo API-n ose sesionin aktiv.',
        }))
      } finally { busy = false }
    }
    void poll()
    const interval = window.setInterval(() => { void poll() }, 1800)
    return () => { active = false; window.clearInterval(interval) }
  }, [scope, cam, sid])

  const shown = scope && history?.key === scope ? history : null
  const alerts = shown?.events ?? []
  return (
    <section className="skyguard-advisory-panel" aria-label="Sinjalizimet inteligjente live">
      <header className="skyguard-advisory-header">
        <div><h2>LIVE MODEL ADVISORIES</h2><p>Sinjalizime për shqyrtim, jo dronë të konfirmuar fizikisht.</p></div>
        <span className="skyguard-advisory-count">{alerts.length} sinjalizime</span>
      </header>
      {!connected && <p role="status">API e shkëputur; vlerësimi i sinjalizimeve u ndal.</p>}
      {connected && liveStatus.state !== 'running' && <p role="status">Nis kamerën për sinjalizimet live.</p>}
      {connected && liveStatus.state === 'running' && !ready && <p role="status">Kërkohet kamerë e regjistruar dhe session_id i vlefshëm.</p>}
      {ready && !shown && <p role="status">Duke marrë vëzhgimet nga motori i alarmeve…</p>}
      {ready && shown?.error && <p role="alert">{shown.error}</p>}
      {ready && shown && alerts.length === 0 && !shown.error && <p role="status">Nuk ka sinjalizime në frame-et e vlerësuara.</p>}
      {ready && shown && <p className="skyguard-advisory-info">Sesioni: {sid?.slice(0, 8)}… · Frame: {shown.frame < 0 ? '—' : shown.frame} · Historik lokal</p>}
      {ready && alerts.length > 0 && <ol className="skyguard-advisory-events" aria-label="Lista e sinjalizimeve">
        {alerts.map(event => <li key={event.event_id} className="skyguard-advisory-event">
          <strong>Shqyrtim operatori · ID {event.track_id}</strong><span>Vëmendje</span>
          <p>Modeli sugjeron dron · {(event.confidence * 100).toFixed(0)}% besueshmëri modeli</p>
          <small>Frame {event.frame_index} · Ngjarja {event.event_id.slice(0, 12)}…</small>
        </li>)}
      </ol>}
      <p role="note" className="skyguard-advisory-disclaimer">
        Vlerësimi përdor vetëm frame-et e kampionuara nga API-ja periodike, jo të gjitha frame-et e kamerës.
        Kjo nuk vërteton identitet fizik, GPS, distancë, apo shkelje të perimetrit.
        Historia nuk ruhet në server dhe nuk ka veprime automatike.
      </p>
    </section>
  )
}
