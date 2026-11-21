import { useEffect, useMemo, useState } from 'react'
import { liveCameraApi, type LiveStatus } from './api/liveCamera'
import { parseLiveMapScene, type LiveMapScene, type LiveMapSceneRequest, type NorthReference } from './api/liveMapScene'
import './LiveMapScenePanel.css'

interface Props {
  liveStatus: LiveStatus
  connected: boolean
}
const SESSION_PATTERN = /^[0-9a-f]{32}$/
const CAMERA_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/
const WIDTH = 760
const HEIGHT = 340

function validIdentity(status: LiveStatus): status is LiveStatus & {
  camera_id: string
  session_id: string
  frame_index: number
} {
  return status.state === 'running' && typeof status.camera_id === 'string' &&
    CAMERA_PATTERN.test(status.camera_id) && typeof status.session_id === 'string' &&
    SESSION_PATTERN.test(status.session_id) && typeof status.frame_index === 'number' &&
    Number.isInteger(status.frame_index) && status.frame_index >= 0
}

export default function LiveMapScenePanel({ liveStatus, connected }: Props) {
  const [northReference, setNorthReference] = useState<NorthReference>('magnetic_north')
  const [scene, setScene] = useState<LiveMapScene | null>(null)
  const [error, setError] = useState<{ cameraId: string; sessionId: string; northReference: NorthReference; message: string } | null>(null)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ east: 0, north: 0 })
  const cameraId = liveStatus.camera_id
  const sessionId = liveStatus.session_id
  const ready = connected && validIdentity(liveStatus)

  useEffect(() => {
    if (!ready || !cameraId || !sessionId) return
    let current = true
    let inFlight = false
    const load = async () => {
      if (inFlight) return
      inFlight = true
      try {
        // Atomic snapshot is selected INSIDE the backend, avoiding the race
        // between GET /status and a POST that guesses the current frame.
        const response = await liveCameraApi.mapSceneLatest(northReference)
        if (!current) return
        if (!Number.isInteger(response.frame_index) || response.frame_index < 0) {
          throw new Error('Invalid frame identity')
        }
        const request: LiveMapSceneRequest = {
          camera_id: cameraId,
          expected_session_id: sessionId,
          expected_frame_index: response.frame_index,
          north_reference: northReference,
        }
        const result = parseLiveMapScene(response, request)
        setScene(previous => previous &&
          previous.session_id === sessionId &&
          previous.camera_id === cameraId &&
          previous.frame_index > result.frame_index ? previous : result)
        setError(null)
      } catch {
        if (current) {
          setError({ cameraId, sessionId, northReference, message: 'Skena nuk është e disponueshme. Kontrollo regjistrimin dhe kalibrimin e kamerës.' })
        }
      } finally {
        inFlight = false
      }
    }
    // Scene and errors are scoped by identity below. No synchronous effect resets.
    void load()
    const handle = window.setInterval(() => { void load() }, 1500)
    return () => { current = false; window.clearInterval(handle) }
  }, [ready, cameraId, sessionId, northReference])

  // Only a map from this exact camera/session can be rendered. Its frame
  // can lag the status poll: it is still internally synchronized by backend.
  const visibleScene = ready && scene?.camera_id === cameraId &&
    scene?.session_id === sessionId && scene?.north_reference === northReference ? scene : null
  const visibleError = ready && error?.cameraId === cameraId &&
    error?.sessionId === sessionId && error?.northReference === northReference ? error.message : null
  const loading = ready && !visibleScene && !visibleError

  const projected = useMemo(() => {
    if (!visibleScene?.camera_location) return null
    const location = visibleScene.camera_location
    const latitudeRadians = location.latitude * Math.PI / 180
    const scale = 1.55 / zoom // metres per SVG pixel; purely schematic
    const project = ([lon, lat]: [number, number]) => ({
      x: WIDTH / 2 + ((lon - location.longitude) * Math.cos(latitudeRadians) * 111320 + offset.east) / scale,
      y: HEIGHT / 2 - ((lat - location.latitude) * 111320 + offset.north) / scale,
    })
    const guides = visibleScene.geojson.features
      .filter(feature => feature.properties.kind === 'bearing_guide')
      .map(feature => {
        if (feature.geometry.type !== 'LineString' || feature.properties.kind !== 'bearing_guide') return null
        return { from: project(feature.geometry.coordinates[0]), to: project(feature.geometry.coordinates[1]), id: feature.properties.detection_index }
      })
      .filter((guide): guide is NonNullable<typeof guide> => guide !== null)
    return { camera: project([location.longitude, location.latitude]), guides, scale }
  }, [visibleScene, zoom, offset])

  const observations = visibleScene?.observations ?? []
  return (
    <section className="skyguard-map-scene" aria-label="SkyGuard mapa e drejtimeve live">
      <div className="skyguard-map-scene-header">
        <div>
          <h2>LIVE GEO SCENE</h2>
          <p>Koordinatat e kamerës dhe drejtimet e detektimeve — jo pozicionet e dronëve.</p>
        </div>
        <label>
          Referenca e veriut
          <select aria-label="Referenca e veriut" value={northReference} onChange={e => setNorthReference(e.target.value as NorthReference)}>
            <option value="magnetic_north">Veriu magnetik (pa vija)</option>
            <option value="true_north">Veriu gjeografik (deklarohet manualisht)</option>
          </select>
        </label>
      </div>
      {!connected && <p role="status">API e shkëputur. Harta live nuk është aktive.</p>}
      {connected && liveStatus.state !== 'running' && <p role="status">Nis kamerën për të marrë të dhëna live.</p>}
      {connected && liveStatus.state === 'running' && !validIdentity(liveStatus) &&
        <p role="status">Kamera duhet të jetë e regjistruar me camera_id dhe session_id për skenën gjeografike.</p>}
      {ready && loading && !visibleScene && <p role="status">Duke sinkronizuar skenën me frame-in aktiv…</p>}
      {ready && visibleError && !visibleScene && !loading && <p role="alert">{visibleError}</p>}
      {visibleScene && (
        <>
          <div className="skyguard-map-scene-status">
            <span>Kamera: {visibleScene.camera_id}</span>
            <span>Frame i skenës: {visibleScene.frame_index}</span>
            <span>Detektime: {observations.length}</span>
            <span>Kalibrimi fizik: i paverifikuar</span>
          </div>
          {!visibleScene.camera_location ? (
            <div className="skyguard-map-scene-empty" role="status">Nuk ka koordinata të deklaruara për kamerën. Nuk mund të vizatojmë drejtimet mbi koordinata gjeografike.</div>
          ) : (
            <div className="skyguard-map-scene-canvas">
              <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Skemë koordinatash e kamerës dhe vijave udhëzuese të drejtimit; nuk përmban GPS të dronëve">
                <rect x="0" y="0" width={WIDTH} height={HEIGHT} className="skyguard-map-scene-background" />
                {Array.from({ length: 19 }, (_, index) => {
                  const x = index * 45
                  return <line key={`v-${index}`} x1={x} x2={x} y1={0} y2={HEIGHT} className="skyguard-map-scene-grid" />
                })}
                {Array.from({ length: 9 }, (_, index) => {
                  const y = index * 45
                  return <line key={`h-${index}`} x1={0} x2={WIDTH} y1={y} y2={y} className="skyguard-map-scene-grid" />
                })}
                <text x={WIDTH - 34} y={25} className="skyguard-map-scene-north">N ↑</text>
                {projected?.guides.map(guide => (
                  <line key={guide.id} x1={guide.from.x} y1={guide.from.y} x2={guide.to.x} y2={guide.to.y} className="skyguard-map-scene-guide" />
                ))}
                {projected && <circle cx={projected.camera.x} cy={projected.camera.y} r={8} className="skyguard-map-scene-camera" />}
                {projected && <text x={projected.camera.x + 14} y={projected.camera.y - 10} className="skyguard-map-scene-camera-label">CAM</text>}
                {projected && <text x={15} y={HEIGHT - 14} className="skyguard-map-scene-scale">100 px ≈ {(100 * projected.scale).toFixed(0)} m (skematik)</text>}
              </svg>
              <div className="skyguard-map-scene-controls" aria-label="Kontrollet e hartës">
                <button type="button" aria-label="Zmadhoni hartën" onClick={() => setZoom(value => Math.min(8, value * 1.5))}>+</button>
                <button type="button" aria-label="Zvogëloni hartën" onClick={() => setZoom(value => Math.max(0.5, value / 1.5))}>−</button>
                <button type="button" aria-label="Lëvizni hartën në veri" onClick={() => setOffset(value => ({ ...value, north: value.north - 60 }))}>↑</button>
                <button type="button" aria-label="Lëvizni hartën në jug" onClick={() => setOffset(value => ({ ...value, north: value.north + 60 }))}>↓</button>
                <button type="button" aria-label="Lëvizni hartën në perëndim" onClick={() => setOffset(value => ({ ...value, east: value.east + 60 }))}>←</button>
                <button type="button" aria-label="Lëvizni hartën në lindje" onClick={() => setOffset(value => ({ ...value, east: value.east - 60 }))}>→</button>
                <button type="button" aria-label="Riktheni hartën" onClick={() => { setZoom(1); setOffset({ east: 0, north: 0 }) }}>Reset</button>
              </div>
            </div>
          )}
          <div className="skyguard-map-scene-disclaimer" role="note">
            <strong>Vetëm vizualizim:</strong> vijat janë 200 m udhëzuese, jo distancë e matur dhe jo pozicion GPS i objektit. Pozicioni i kamerës vjen nga regjistri aktual, jo nga një matje e verifikuar në momentin e frame-it.
          </div>
          {observations.length > 0 && <div className="skyguard-map-scene-observations" aria-label="Drejtimet e objekteve">
            {observations.slice(0, 20).map(item => <span key={item.detection_index}>
              {item.object_class} #{item.detection_index}: {item.azimuth_degrees === null ? 'azimut pa të dhëna' : `${item.azimuth_degrees.toFixed(1)}°`}, elev. {item.elevation_degrees.toFixed(1)}°
            </span>)}
            {observations.length > 20 && <span>+{observations.length - 20} detektime të tjera</span>}
          </div>}
        </>
      )}
    </section>
  )
}
