import AirspaceMap from './AirspaceMap'
import LiveMapScenePanel from './LiveMapScenePanel'
import LiveTrackHistoryPanel from './LiveTrackHistoryPanel'
import LiveAdvisoryPanel from './LiveAdvisoryPanel'
import TrajectoryOverlay from './TrajectoryOverlay'
import { useSynchronizedFrame } from './useSynchronizedFrame'
import CameraManager, { type CameraSelection } from './CameraManager'
import { useCallback, useEffect, useRef, useState } from 'react'
import { liveCameraApi, type LiveStatus, type LiveCameraSource } from './api/liveCamera'
import './App.css'

type Detection = {
  class: string
  confidence: number
  bbox: {
    x: number
    y: number
    width: number
    height: number
  }
}

type Track = {
  track_id: number
  class: string
  confidence: number
  state: string
  frame_index: number
}

type EventEntry = {
  id: string
  time: string
  message: string
}

const EMPTY_STATUS: LiveStatus = {
  state: 'idle',
  frame_index: null,
  detections: [],
  tracks: [],
  processing_seconds: null,
  error: null,
}

function getDetections(status: LiveStatus): Detection[] {
  return Array.isArray(status.detections)
    ? status.detections as Detection[]
    : []
}

function getTracks(status: LiveStatus): Track[] {
  return Array.isArray(status.tracks)
    ? status.tracks as Track[]
    : []
}

export default function App() {
  const [status, setStatus] = useState<LiveStatus>(EMPTY_STATUS)
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const [cameraIndex, setCameraIndex] = useState(0)
  const [sourceType, setSourceType] = useState<'usb' | 'rtsp'>('usb')
  const [rtspUrl, setRtspUrl] = useState('')
  const [confidence, setConfidence] = useState(0.35)
  const [fusionEnabled, setFusionEnabled] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<EventEntry[]>([])
  const [streamVersion, setStreamVersion] = useState(0)  const [showTrajectories, setShowTrajectories] = useState(false)
  const [streamDimensions, setStreamDimensions] = useState<
    { width: number; height: number; version: number } | null
  >(null)


  const synchronized = useSynchronizedFrame(
    showTrajectories && status.state === 'running' && !busy,
    streamVersion,
  )

  const seenTracks = useRef(new Set<number>())
  const sessionGeneration = useRef(0)
  const operationInProgress = useRef(false)

  const refresh = useCallback(async (expectedGeneration?: number) => {
    const generation = expectedGeneration ?? sessionGeneration.current
    const isOperationRefresh = expectedGeneration !== undefined

    if (operationInProgress.current && !isOperationRefresh) return
    try {
      const result = await liveCameraApi.status()

      if (
        generation !== sessionGeneration.current ||
        (operationInProgress.current && !isOperationRefresh)
      ) return

      setStatus(result)
      setConnected(true)

      const tracks = getTracks(result)
      const newEvents: EventEntry[] = []

      for (const track of tracks) {
        if (seenTracks.current.has(track.track_id)) continue

        seenTracks.current.add(track.track_id)

        newEvents.push({
          id: `${track.track_id}-${Date.now()}`,
          time: new Date().toLocaleTimeString(),
          message: `${track.class.toUpperCase()} #${track.track_id} u identifikua (${Math.round(track.confidence * 100)}%)`,
        })
      }

      if (newEvents.length > 0) {
        setEvents(previous => [...newEvents, ...previous].slice(0, 100))
      }
    } catch (err) {
      if (
        generation !== sessionGeneration.current ||
        (operationInProgress.current && !isOperationRefresh)
      ) return
      setConnected(false)
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const poll = async () => {
      try {
        await refresh()
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(poll, 1000)
        }
      }
    }

    void poll()

    return () => {
      cancelled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [refresh])

  const selectManagedCamera = (source: CameraSelection) => {
    if (source.source_type === 'usb') {
      setSourceType('usb')
      setCameraIndex(source.source_location)
    } else {
      setSourceType('rtsp')
      setRtspUrl(source.source_location)
    }
    setError(null)
  }

  const startCamera = async () => {
    if (operationInProgress.current) return

    operationInProgress.current = true
    sessionGeneration.current += 1
    const operationGeneration = sessionGeneration.current
    setStreamVersion(previous => previous + 1)
    setBusy(true)
    setError(null)

    try {
      seenTracks.current.clear()
      setEvents([])

      let source: LiveCameraSource

      if (sourceType === 'rtsp') {
        const address = rtspUrl.trim()

        if (!/^rtsp:\/\/[^\s/]+(?:\/[^\s]*)?$/i.test(address)) {
          throw new Error('Enter a valid rtsp:// camera address.')
        }

        source = {
          source_type: 'rtsp',
          source_location: address,
        }
      } else {
        source = {
          source_type: 'usb',
          source_location: cameraIndex,
        }
      }

      const result = await liveCameraApi.start(
        cameraIndex,
        confidence,
        source,
        fusionEnabled,
      )

      setStatus(result)
      await refresh(operationGeneration)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      operationInProgress.current = false
      setBusy(false)
    }
  }

  const stopCamera = async () => {
    if (operationInProgress.current) return

    operationInProgress.current = true
    sessionGeneration.current += 1
    const operationGeneration = sessionGeneration.current
    setStreamVersion(previous => previous + 1)
    setBusy(true)
    setError(null)
    seenTracks.current.clear()
    setEvents([])

    try {
      const result = await liveCameraApi.stop()
      setStatus(result)
      await refresh(operationGeneration)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      operationInProgress.current = false
      setBusy(false)
    }
  }

  const detections = getDetections(status)
  const tracks = getTracks(status)  const trajectories = Array.isArray(status.trajectories)
    ? status.trajectories
    : []

  // Only confirmed tracks observed in the current API frame
  // qualify for the live overlay. Historical data remains intact.
  const activeTrackIds = new Set(
    tracks
      .filter(
        track =>
          track.state === 'confirmed' &&
          track.frame_index === status.frame_index,
      )
      .map(track => track.track_id),
  )

  const activeTrajectories = trajectories.filter(trajectory => {
    if (!activeTrackIds.has(trajectory.track_id)) {
      return false
    }

    if (
      status.frame_index == null ||
      !Array.isArray(trajectory.points) ||
      trajectory.points.length < 2
    ) {
      return false
    }

    const latest = trajectory.points[trajectory.points.length - 1]

    return latest.frame_index === status.frame_index
  })

  const active = status.state === 'running'
  const sessionActive = active || status.state === 'starting'

  const drones = detections.filter(item => item.class === 'drone')
  const birds = detections.filter(item => item.class === 'bird')
  const unknownCandidates = detections.filter(item => item.class === 'unknown')
  const ambiguousTracks = Array.isArray(status.ambiguous_tracks)
    ? status.ambiguous_tracks
    : []

  return (
    <div className="skyguard">
      <header className="header">
        <div className="brand">
          <h1>SKY<span>GUARD</span> FUSION</h1>
          <p>Ndërgjegjësimi situacional i hapësirës ajrore · AI Camera Monitoring</p>
        </div>

        <div className="header-stats">
          <div className="stat">
            <span>Dronë të zbuluar</span>
            <strong>{drones.length}</strong>
          </div>

          <div className="stat">
            <span>Zogj</span>
            <strong>{birds.length}</strong>
          </div>

          <div className="stat">
            <span>UNKNOWN candidates</span>
            <strong>{unknownCandidates.length}</strong>
          </div>

          <div className="stat">
            <span>Ambiguous identities</span>
            <strong>{ambiguousTracks.length}</strong>
          </div>

          <div className="stat">
            <span>Objekte në tracking</span>
            <strong>{tracks.length}</strong>
          </div>

          <div className={`live-indicator ${active ? 'active' : ''}`}>
            <span className="live-dot" />
            {active ? 'LIVE' : status.state.toUpperCase()}
          </div>
        </div>
      </header>

      <main className="workspace">
        <section className="camera-section">
          <div className="camera-toolbar">
            <div className="toolbar-title">
              <span className="camera-icon">◉</span>
              <span>LIVE CAMERA · YOLO DETECTION</span>
            </div>

            <div className="toolbar-status">
              {connected ? 'API CONNECTED' : 'API DISCONNECTED'}
            </div>
          </div>

          <div className="camera-view">
            {sessionActive ? (
              <>
                <img
                key={streamVersion}
                src={
                  showTrajectories
                    ? synchronized.frame
                      ? `data:image/jpeg;base64,${synchronized.frame.image_base64}`
                      : undefined
                    : liveCameraApi.streamUrl
                }
                alt="SkyGuard live camera with AI detections"
                className="camera-stream"
                onLoad={event => {
                  const image = event.currentTarget
                  const width = image.naturalWidth
                  const height = image.naturalHeight
                  if (width > 0 && height > 0) {
                    setStreamDimensions(previous =>
                      previous?.width === width &&
                      previous.height === height &&
                      previous.version === streamVersion
                        ? previous
                        : { width, height, version: streamVersion },
                    )
                  }
                }}
              />

              {showTrajectories &&
                streamDimensions?.version === streamVersion && (
                <TrajectoryOverlay
                  trajectories={
                    showTrajectories
                      ? synchronized.frame?.trajectories.filter(trajectory => {
                          const frame = synchronized.frame
                          if (!frame) return false

                          const confirmed = frame.tracks.some(
                            track =>
                              track.track_id === trajectory.track_id &&
                              track.state === 'confirmed' &&
                              track.frame_index === frame.frame_index,
                          )

                          const points = trajectory.points
                          return (
                            confirmed &&
                            Array.isArray(points) &&
                            points.length >= 2 &&
                            points[points.length - 1].frame_index ===
                              frame.frame_index
                          )
                        }) ?? []
                      : activeTrajectories
                  }
                  imageWidth={streamDimensions.width}
                  imageHeight={streamDimensions.height}
                />
              )}
              </>
            ) : (
              <div className="camera-offline">
                <div className="offline-icon">◎</div>
                <h2>KAMERA JOAKTIVE</h2>
                <p>Nis kamerën për të aktivizuar detektimin dhe gjurmimin.</p>
              </div>
            )}

            <div className="camera-overlay top-left">
              SKYGUARD / CAM-{cameraIndex.toString().padStart(2, '0')}
            </div>

            <div className="camera-overlay bottom-left">
              FRAME: {showTrajectories
                ? synchronized.frame?.frame_index ?? '?'
                : status.frame_index ?? '?'}
            </div>

            <div className="camera-overlay bottom-right">
              AI INFERENCE: {status.processing_seconds == null
                ? '—'
                : `${(status.processing_seconds * 1000).toFixed(0)} ms`}
            </div>
          </div>

          <div className="trajectory-toggle">
            <label>
              <input
                type="checkbox"
                checked={showTrajectories}
                onChange={event => setShowTrajectories(event.target.checked)}
                disabled={!sessionActive}
              />
              SHOW TRAJECTORIES
            </label>
            <span>
              Image-space trajectories synchronized with the displayed JPEG frame.
            </span>
          </div>

          <div className="camera-controls">
            <div className="control-field">
              <label htmlFor="source-type">BURIMI VIDEO</label>
              <select
                id="source-type"
                value={sourceType}
                disabled={sessionActive || busy}
                onChange={event =>
                  setSourceType(event.target.value as 'usb' | 'rtsp')
                }
              >
                <option value="usb">USB / Laptop Camera</option>
                <option value="rtsp">IP / RTSP Camera</option>
              </select>
            </div>

            {sourceType === 'rtsp' && (
              <div className="control-field">
                <label htmlFor="rtsp-url">RTSP ADDRESS</label>
                <input
                  id="rtsp-url"
                  type="text"
                  value={rtspUrl}
                  disabled={sessionActive || busy}
                  onChange={event => setRtspUrl(event.target.value)}
                  placeholder="rtsp://camera-host/stream"
                  autoComplete="off"
                />
              </div>
            )}

            {sourceType === 'usb' && (
            <div className="control-field">
              <label htmlFor="camera-index">KAMERA</label>
              <select
                id="camera-index"
                value={cameraIndex}
                disabled={sessionActive || busy}
                onChange={event => setCameraIndex(Number(event.target.value))}
              >
                {[0, 1, 2, 3, 4].map(index => (
                  <option value={index} key={index}>Camera {index}</option>
                ))}
              </select>
            </div>
            )}

            <div className="control-field confidence-field">
              <label htmlFor="confidence">
                CONFIDENCE · {Math.round(confidence * 100)}%
              </label>
              <input
                id="confidence"
                type="range"
                min="0.01"
                max="1"
                step="0.01"
                value={confidence}
                disabled={sessionActive || busy}
                onChange={event => setConfidence(Number(event.target.value))}
              />
            </div>

            <div className="control-field">
              <label htmlFor="fusion-enabled">
                FUSION DETECTION
              </label>
              <label
                htmlFor="fusion-enabled"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <input
                  id="fusion-enabled"
                  type="checkbox"
                  checked={fusionEnabled}
                  disabled={sessionActive || busy}
                  onChange={event =>
                    setFusionEnabled(event.target.checked)
                  }
                />
                <span>
                  YOLO + Motion Candidates
                </span>
              </label>
              <small>
                Motion candidates are unverified and may include
                camera movement or non-airborne objects.
              </small>
            </div>

            <button
              className="button-start"
              disabled={sessionActive || busy || !connected}
              onClick={() => void startCamera()}
            >
              ▶ START CAMERA
            </button>

            <button
              className="button-stop"
              disabled={!sessionActive || busy || !connected}
              onClick={() => void stopCamera()}
            >
              ■ STOP
            </button>
          </div>

          {(error || status.error) && (
            <div className="error-box">{error || status.error}</div>
          )}

          <div className="detections-section">
            <div className="section-heading">
              <h2>DETEKTIMET AKTUALE</h2>
              <span>{detections.length} OBJEKTE</span>
            </div>

            {detections.length === 0 ? (
              <div className="empty-message">Nuk ka detektime aktive.</div>
            ) : (
              <div className="detection-list">
                {detections.map((item, index) => (
                  <div className="detection-item" key={index}>
                    <span className={`object-class ${item.class}`}>
                      {item.class.toUpperCase()}
                    </span>
                    <span>{Math.round(item.confidence * 100)}%</span>
                    <span>
                      X:{Math.round(item.bbox.x)} Y:{Math.round(item.bbox.y)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        <aside className="sidebar">
          <section className="sidebar-panel">
            <div className="section-heading">
              <h2>GJURMËT E OBJEKTEVE</h2>
              <span>{tracks.length}</span>
            </div>

            {tracks.length === 0 ? (
              <div className="empty-message">Nuk ka objekte në tracking.</div>
            ) : (
              <div className="track-list">
                {tracks.map(track => (
                  <div className="track-item" key={track.track_id}>
                    <span className="track-id">
                      {track.class.toUpperCase()}-{String(track.track_id).padStart(3, '0')}
                    </span>
                    <span>{track.state}</span>
                    <strong>{Math.round(track.confidence * 100)}%</strong>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="sidebar-panel">
            <div className="section-heading">
              <h2>AMBIGUOUS IDENTITIES</h2>
              <span>{ambiguousTracks.length}</span>
            </div>

            {ambiguousTracks.length === 0 ? (
              <div className="empty-message">
                No ambiguous identities in the current frame.
              </div>
            ) : (
              <div className="track-list">
                {ambiguousTracks.map((candidate, index) => (
                  <div className="track-item" key={index}>
                    <span className="track-id">
                      {String(candidate.class ?? 'unknown').toUpperCase()}
                    </span>
                    <span>
                      {String(candidate.state ?? 'ambiguous')}
                    </span>
                    <strong>
                      Possible IDs: {
                        Array.isArray(candidate.candidate_track_ids)
                          ? candidate.candidate_track_ids.join(', ') || 'none'
                          : 'not provided'
                      }
                    </strong>
                  </div>
                ))}
              </div>
            )}

            <div className="empty-message">
              Ambiguous matches are hypotheses, not confirmed track IDs.
            </div>
          </section>

          <section className="sidebar-panel">
            <div className="section-heading">
              <h2>TRAJECTORY HISTORY (IMAGE COORDINATES)</h2>
              <span>{trajectories.length}</span>
            </div>

            {trajectories.length === 0 ? (
              <div className="empty-message">
                No confirmed trajectory history available.
              </div>
            ) : (
              <div className="track-list">
                {trajectories.slice(0, 20).map(trajectory => {
                  const points = Array.isArray(trajectory.points)
                    ? trajectory.points
                    : []
                  const latest = points[points.length - 1]
                  const current = tracks.some(
                    track => track.track_id === trajectory.track_id,
                  )

                  return (
                    <div className="track-item" key={trajectory.track_id}>
                      <div>
                        <strong>
                          {trajectory.object_class.toUpperCase()}
                          {' #'}{trajectory.track_id}
                        </strong>
                        <div>
                          {current ? 'Current track' : 'Historical track'}
                        </div>
                        <div>
                          {points.length} recorded points
                        </div>
                        {latest && (
                          <>
                            <div>Frame {latest.frame_index}</div>
                            <div>
                              X: {Number.isFinite(latest.x_px)
                                ? latest.x_px.toFixed(1)
                                : 'N/A'} px
                              {' | '}
                              Y: {Number.isFinite(latest.y_px)
                                ? latest.y_px.toFixed(1)
                                : 'N/A'} px
                            </div>
                            <div>
                              Speed: {
                                latest.speed_px_s !== null &&
                                Number.isFinite(latest.speed_px_s)
                                  ? latest.speed_px_s.toFixed(2) + ' px/s'
                                  : 'N/A'
                              }
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {trajectories.length > 20 && (
              <div className="empty-message">
                Showing 20 of {trajectories.length} trajectories.
              </div>
            )}

            <div className="empty-message">
              Positions and speeds are measured in image pixels.
              They are not GPS coordinates, physical distances,
              or calibrated real-world velocities.
              Historical trajectories may remain after an object
              leaves the current frame.
            </div>
          </section>

          <section className="sidebar-panel events-panel">
            <div className="section-heading">
              <h2>REGJISTRI I NGJARJEVE</h2>
              <span>{events.length}</span>
            </div>

            {events.length === 0 ? (
              <div className="empty-message">Në pritje të detektimeve...</div>
            ) : (
              <div className="events-list">
                {events.map(event => (
                  <div className="event-item" key={event.id}>
                    <span>{event.time}</span> — {event.message}
                  </div>
                ))}
              </div>
            )}
          </section>

          <div className="system-note">
            DRONE/BIRD classifications come from YOLO.
            With Fusion enabled, UNKNOWN denotes unverified
            motion candidates. Camera motion and ground objects
            can produce false candidates. Object GPS coordinates
            and measured ranges are not available in this version.
          </div>

          <div className="system-note">
            Të dhënat vijnë nga kamera dhe modeli YOLO.
            Distancat dhe koordinatat gjeografike nuk llogariten në këtë version.
          </div>
        </aside>
      </main>

      <section className="airspace-map-section">
        <CameraManager onSelectCamera={selectManagedCamera} disabled={sessionActive || busy} />
      </section>

      <section className="airspace-map-section">
        <AirspaceMap liveStatus={status} connected={connected} />
        <LiveMapScenePanel liveStatus={status} connected={connected} />
        <LiveTrackHistoryPanel liveStatus={status} connected={connected} />
        <LiveAdvisoryPanel liveStatus={status} connected={connected} />
      </section>

      <footer className="footer">
        SKYGUARD FUSION · AI-POWERED VISUAL AIRSPACE MONITORING
      </footer>
    </div>
  )
}
