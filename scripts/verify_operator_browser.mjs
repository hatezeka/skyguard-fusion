import { spawn } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const [browser, url, image, video, mode] = process.argv.slice(2)
const profile = await mkdtemp(join(tmpdir(), 'skyguard-edge-'))
const edge = spawn(browser, [
  '--headless=new', '--no-first-run', '--disable-background-networking',
  '--enable-unsafe-swiftshader', '--remote-debugging-port=0',
  '--user-data-dir=' + profile, 'about:blank',
], { windowsHide: true, stdio: 'ignore' })
let socket
async function waitFor(check, description, timeout = 30000) {
  const deadline = Date.now() + timeout
  do {
    const value = await check()
    if (value) return value
    await new Promise(resolve => setTimeout(resolve, 300))
  } while (Date.now() < deadline)
  throw new Error('Timed out waiting for ' + description)
}

try {
  const port = await waitFor(async () => {
    try { return (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split(/\r?\n/)[0] }
    catch { return null }
  }, 'Edge devtools')
  const pages = await (await fetch('http://127.0.0.1:' + port + '/json/list')).json()
  socket = new WebSocket(pages.find(page => page.type === 'page').webSocketDebuggerUrl)
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true })
    socket.addEventListener('error', reject, { once: true })
  })
  let id = 0
  const pending = new Map()
  const tiles = new Map()
  const tileRequests = new Map()
  socket.addEventListener('message', event => {
    const msg = JSON.parse(event.data)
    if (msg.method === 'Network.responseReceived') {
      const { requestId, response } = msg.params
      const address = new URL(response.url)
      if (address.hostname === 'tiles.openfreemap.org' && /\.(pbf|mvt)$/.test(address.pathname) && response.status === 200) {
        tileRequests.set(requestId, address.pathname)
      }
    }
    if (msg.method === 'Network.loadingFinished' && tileRequests.has(msg.params.requestId)) {
      const name = tileRequests.get(msg.params.requestId)
      tiles.set(name, msg.params.encodedDataLength)
    }
    const entry = pending.get(msg.id)
    if (!entry) return
    pending.delete(msg.id)
    if (msg.error) entry.reject(new Error(msg.error.message))
    else entry.resolve(msg.result)
  })
  function call(method, params = {}) {
    return new Promise((resolve, reject) => {
      pending.set(++id, { resolve, reject })
      socket.send(JSON.stringify({ id, method, params }))
    })
  }
  async function evaluate(expression) {
    const result = await call('Runtime.evaluate', { expression, returnByValue: true })
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text)
    return result.result.value
  }
  async function click(label) {
    return waitFor(() => evaluate(`(() => {
      const button = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === ${JSON.stringify(label)} && !b.disabled)
      if (!button) return false
      button.click()
      return true
    })()`), label)
  }
  if (mode === '--camera-coverage') {
    // Browser-only evidence fixtures. The live service and database are read-only throughout.
    const camera = {id:'coverage-test-camera', name:'Coverage test fixture', source_kind:'USB',
      latitude:42.6, longitude:21.1, calibration_ref:'test-calibration'}
    const benchmark = {id:'coverage-test', validation_level:'PHYSICAL_VERIFIED',metrics:{camera_coverage_v1:{
      camera_id:camera.id, latitude:camera.latitude, longitude:camera.longitude,
      calibration_ref:camera.calibration_ref, evidence_ref:'test-fixture-only', position_source:'Test source',
      position_uncertainty_m:1.25, fixed_camera:true, heading_deg:90, horizontal_fov_deg:50,
      trials:[{latitude:42.6, longitude:21.101, horizontal_m:120, bearing_deg:90, passed:true}],
    }}}
    const fixtures={'/api/persistence/cameras':[camera],'/api/persistence/benchmarks':[benchmark],
      '/styles/dark':{version:8,sources:{},layers:[{id:'fixture-base',type:'background',paint:{'background-color':'#10212e'}}]}}
    socket.addEventListener('message',event=>{
      const msg=JSON.parse(event.data)
      if(msg.method!=='Fetch.requestPaused')return
      const {request,requestId}=msg.params
      const value=fixtures[new URL(request.url).pathname]
      if(value)void call('Fetch.fulfillRequest',{requestId,responseCode:200,
       responseHeaders:[{name:'Content-Type',value:'application/json'},{name:'Access-Control-Allow-Origin',value:'*'}],body:Buffer.from(JSON.stringify(value)).toString('base64')})
      else void call('Fetch.continueRequest',{requestId})
    })
    await call('Fetch.enable',{patterns:Object.keys(fixtures).map(path=>({urlPattern:'*'+path+'*'}))})
  }
  await call('Network.enable')
  await call('Page.navigate', { url })
  async function connectPersistence() {
    await waitFor(() => evaluate("document.body.innerText.includes('DATABASE CONNECTED')"), 'automatic database connection')
  }
  if (mode === '--geospatial-reference') {
    const result = await waitFor(() => evaluate(`(() => {
      const reference = document.querySelector('[aria-label="Geospatial reference / Camera rig location"]')
      const map = document.querySelector('[aria-label="Interactive real geographic map"] canvas')
      const tactile = [...document.querySelectorAll('h2,h3')].some(el=>el.textContent==='Tactical representation')
      const status = reference?.querySelector('[role="status"]')?.textContent?.trim()
      if(!status?.startsWith('GEO_REFERENCE_') || !map || !tactile) return null
      return {status,geographicCanvas:map.width>0&&map.height>0,tacticalPanel:tactile}
    })()`), 'rig configuration and two map panels', 45000)
    process.stdout.write(JSON.stringify(result))
  } else
  if (mode === '--visible-stereo') {
    await click('Live observation')
    const result = await waitFor(() => evaluate(`(() => {
      const panels = [...document.querySelectorAll('.sg-stereo-grid section')]
      if (panels.length !== 2) return null
      const frames = panels.map(panel => {
        const image = panel.querySelector('img')
        const bounds = image?.getBoundingClientRect()
        return {label:panel.getAttribute('aria-label'), width:image?.naturalWidth, height:image?.naturalHeight,
                visible: Boolean(bounds && bounds.width > 0 && bounds.height > 0)}
      })
      return frames.every(frame => frame.visible && frame.width === 640 && frame.height === 480) ? frames : null
    })()`), 'two decoded stereo images', 65000)
    process.stdout.write(JSON.stringify({frames:result}))
  } else
  if (mode === '--multi-object-image') {
    await click('Live observation')
    const {root} = await call('DOM.getDocument')
    const {nodeId} = await call('DOM.querySelector', {
      nodeId:root.nodeId, selector:'input[aria-label="Choose image or video"]',
    })
    const started = performance.now()
    await call('DOM.setFileInputFiles', {nodeId, files:[image]})
    const result = await waitFor(() => evaluate(`(() => {
      const panel = document.querySelector('[aria-label="ANALYSIS COMPLETE"]')
      const photo = document.querySelector('img[alt="Annotated upload"]')
      if (!panel || !photo?.complete || !photo.naturalWidth) return null
      panel.scrollIntoView({block:'end'})
      return Object.fromEntries([...panel.querySelectorAll('[data-class]')].map(
        el => [el.dataset.class, Number(el.querySelector('dd').textContent)]))
    })()`), 'annotated image and object cards', 150000)
    const latencyMs = Math.round(performance.now() - started)
    const {data} = await call('Page.captureScreenshot', {format:'png'})
    await writeFile(video, Buffer.from(data,'base64'), {flag:'wx'})
    process.stdout.write(JSON.stringify({counts:result, latencyMs, screenshot:video}))
    if(result.DRONE!==1 || result.BIRD!==1 || result.UNKNOWN!==0) process.exitCode=1
  } else
  if (mode === '--camera-coverage') {
    await connectPersistence()
    await click('Dual map views')
    await waitFor(()=>evaluate("!!document.querySelector('.sg-map-coverage-controls select option[value=coverage-test-camera]')"),'camera selector')
    await evaluate(`(() => {
      const select=document.querySelector('.sg-map-coverage-controls select')
      select.value='coverage-test-camera'
      select.dispatchEvent(new Event('change',{bubbles:true}))
    })()`)
    await waitFor(()=>evaluate("!!document.querySelector('.sg-map-sensor') && document.querySelectorAll('.sg-map-ring-label').length===3"),'camera marker and rings')
      .catch(async error=>{
        const diagnostic=await evaluate(`({
          camera:document.querySelector('.sg-map-coverage-controls select')?.value,
          rings:document.querySelectorAll('.sg-map-ring-label').length,
          markers:document.querySelectorAll('.sg-map-sensor').length,
          mapCanvas:!!document.querySelector('.maplibregl-canvas'),
          mapError:document.querySelector('.sg-map-status-error')?.textContent,
        })`)
        throw new Error(error.message+': '+JSON.stringify(diagnostic))
      })
    const status=await evaluate("document.body.innerText.includes('Detektimi u verifikua deri në 120 m në drejtimin e testuar')")
    if(!status)throw new Error('Missing one-direction evidence label')
    const evidence=[]
    for(const [device,width,height] of [['desktop',1440,900],['phone',390,844]]){
      await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:device==='phone'})
      const view=await waitFor(()=>evaluate(`(() => {
        const map=document.querySelector('.sg-maplibre-host').getBoundingClientRect()
        const camera=document.querySelector('.sg-map-sensor').getBoundingClientRect()
        const dx=Math.abs((camera.left+camera.right-map.left-map.right)/2)
        const dy=Math.abs((camera.top+camera.bottom-map.top-map.bottom)/2)
        if(dx>14||dy>14||window.innerWidth!==${width})return null
        return {markerCentered:true, ringLabels:[...document.querySelectorAll('.sg-map-ring-label')].map(e=>e.textContent),
          horizontalOverflow:document.documentElement.scrollWidth>document.documentElement.clientWidth+1}
      })()`),'camera centered and scale visible on '+device)
      if(view.horizontalOverflow||view.ringLabels.some(label=>!label.endsWith(' m')))throw new Error('Coverage viewport overflow or missing units')
      evidence.push({device,...view})
    }
    const initial=await evaluate("document.querySelector('.sg-map-ring-label')?.textContent")
    await evaluate("document.querySelector('.maplibregl-ctrl-zoom-in').click()")
    await evaluate("document.querySelector('.maplibregl-ctrl-zoom-in').click()")
    await waitFor(()=>evaluate(`document.querySelector('.sg-map-ring-label')?.textContent !== ${JSON.stringify(initial)}`),'zoom-dependent scale')
    process.stdout.write(JSON.stringify({ fixtureOnly:true, oneDirectionLabel:true, zoomAdaptive:true, evidence }))
  } else
  if (mode === '--auto-persistence') {
    const databaseConnected = () => evaluate("document.body.innerText.includes('DATABASE CONNECTED')")
    await waitFor(databaseConnected, 'automatic database connection')
    await click('Dual map views')
    const selectedSession = () => evaluate("document.querySelector('.sg-title-actions select')?.value || null")
    const before = await waitFor(selectedSession, 'session selected')
    const resumeUrl = new URL(url)
    resumeUrl.searchParams.set('verify-auto-reload', '1')
    await call('Page.navigate', { url: resumeUrl.href })
    await waitFor(() => evaluate("location.search.includes('verify-auto-reload=1') && !!document.querySelector('.sg-root')"), 'new dashboard document')
    await waitFor(databaseConnected, 'automatic reconnection after refresh')
    const after = await waitFor(selectedSession, 'session restored')
    const retainedView = await evaluate("document.querySelector('main h1')?.textContent === 'Dual map views'")
    if (before !== after || !retainedView) throw new Error('The session or view was not restored')
    const hiddenPages = await evaluate(`[...document.querySelectorAll('nav button')].every(button =>
      !['Scientific validation','Integration settings'].includes(button.textContent.trim()))`)
    await click('Command center')
    const overviewHasUpload = await evaluate("!!document.querySelector('.sg-media-lab')")
    await click('Live observation')
    await waitFor(()=>evaluate("!!document.querySelector('.sg-media-lab input[type=file]')"),'original MediaLab upload')
    if(!hiddenPages||overviewHasUpload)throw new Error('Unexpected navigation or uploader placement')
    process.stdout.write(JSON.stringify({ databaseConnectedAfterReload: true, sessionRestored: true, retainedView,
      removedPagesHidden:true, mediaLabOnLiveObservation:true }))
  } else {
  await connectPersistence()
  if (mode === '--tracking') {
    async function inspectHistory() {
      await click('Object tracking')
      await click('Inspect')
      return waitFor(() => evaluate(`(() => {
        const timeline = document.querySelector('.sg-selected-line')?.textContent ?? ''
        const frames = Number(timeline.match(/([0-9]+) frames returned/)?.[1] ?? 0)
        return frames >= 3 && timeline.includes('DRONE') ? frames : null
      })()`), 'database track history')
    }
    const beforeReload = await inspectHistory()
    const resumeUrl = new URL(url)
    resumeUrl.searchParams.set('verify-tracking-reload', '1')
    await call('Page.navigate', { url: resumeUrl.href })
    await waitFor(() => evaluate("location.search.includes('verify-tracking-reload=1') && !!document.querySelector('.sg-root')"), 'new dashboard document')
    await connectPersistence()
    const afterReload = await inspectHistory()
    const trackingId = await evaluate("document.querySelector('table tbody tr code')?.textContent?.trim()")
    await click('Dual map views')
    const pixelPath = await waitFor(() => evaluate(`(() => {
      const chart = document.querySelector('.sg-pixel-tracks')
      const line = chart?.querySelector('polyline')
      const label = chart?.querySelector('button')?.textContent
      const unavailable = document.body.innerText.includes('Vendndodhja e dronit e padisponueshme')
      if (!line || !label || !unavailable) return null
      return { points: line.getAttribute('points').split(' ').length, label,
        noGeographicDroneMarker: !document.querySelector('.sg-map-drone'), metersShown: !!document.querySelector('.scale-mark') }
    })()`), 'camera pixel trajectory and absence of GPS')
      .catch(async error=>{
        const view=await evaluate(`({
          heading:document.querySelector('main h1')?.textContent,
          session:document.querySelector('select[aria-label="Active recording session"]')?.value,
          chart:document.querySelector('.sg-pixel-tracks')?.textContent,
          statuses:[...document.querySelectorAll('.sg-map-status-info,.sg-map-status-error')].map(e=>e.textContent),
          radar:!!document.querySelector('.sg-angular-radar'),
          text:document.querySelector('main')?.innerText.slice(0,900),
        })`)
        throw new Error(error.message+': '+JSON.stringify(view))
      })
    if (!trackingId || !pixelPath.label.includes(trackingId) || pixelPath.points < 2 ||
        !pixelPath.noGeographicDroneMarker || pixelPath.metersShown) throw new Error('Pixel map integrity check failed')
    process.stdout.write(JSON.stringify({ beforeReload, afterReload, trackingId, pixelPath }))
  } else {
  const viewportEvidence = []
  if (mode === '--responsive') {
    const screens = [['phone', 360, 800], ['tablet', 768, 1024], ['desktop', 1440, 900]]
    const pages = ['Command center','Live observation','Object tracking','Dual map views','Alert management',
      'Evidence ledger','Performance','Camera registry']
    for (const [device, width, height] of screens) {
      await call('Emulation.setDeviceMetricsOverride', { width, height, mobile: width <= 800, deviceScaleFactor: 1 })
      for (const page of pages) {
        if (width <= 800) await evaluate("document.querySelector('[aria-label=\"Open navigation\"]').click()")
        await click(page)
        const result = await evaluate(`({
          viewport: document.documentElement.clientWidth,
          content: document.documentElement.scrollWidth,
          heading: document.querySelector('main h1')?.textContent,
          scrollableTables: [...document.querySelectorAll('.sg-table-scroll')].every(t=>t.scrollWidth<=t.clientWidth || getComputedStyle(t).overflowX==='auto'),
          primaryControlsFit: [...document.querySelectorAll('main button, main input:not([hidden]), main select')].filter(e=>e.getClientRects().length && !e.closest('.sg-table-scroll')).every(e=>{
            const r=e.getBoundingClientRect()
            return r.left>=-1 && r.right<=document.documentElement.clientWidth+1
          })
        })`)
        viewportEvidence.push({ device, page, ...result })
      }
    }
    await call('Emulation.clearDeviceMetricsOverride')
  }
  await click('Dual map views')
  await waitFor(() => evaluate("!!document.querySelector('[aria-label=\"Sensor-relative tactical positions\"]')"), 'tactical map')
  const geographicCanvas = await waitFor(() => evaluate("!!document.querySelector('canvas.maplibregl-canvas')"), 'geographic map canvas', 12000).catch(() => false)
  const geographicTiles = await waitFor(() => tiles.size || null, 'successful OpenFreeMap vector tile', 20000)
    .catch(() => 0)
  const mapError = await evaluate("document.querySelector('.sg-map-status-error')?.textContent ?? null")
  const mapEvidence = { databaseConnected: true, tacticalMap: true, geographicCanvas, geographicTiles, mapError }
  if (mode === '--map-only' || mode === '--responsive') {
    process.stdout.write(JSON.stringify({ ...mapEvidence, viewports: viewportEvidence }))
  } else {
  await click('Live observation')
  const previews = []
  const uploadTimeToDisplayMs = {}
  for (const [kind, file, selector] of [
    ['IMAGE', image, 'img[alt="Annotated upload"]'],
    ['VIDEO', video, 'video.sg-media-output'],
  ]) {
    const started = performance.now()
    const { root } = await call('DOM.getDocument')
    const { nodeId } = await call('DOM.querySelector', { nodeId: root.nodeId, selector: 'input[aria-label="Choose image or video"]' })
    await call('DOM.setFileInputFiles', { nodeId, files: [file] })
    await waitFor(() => evaluate(`[...document.querySelectorAll('h3')].some(node => node.textContent === 'Job: DONE') &&
      !!document.querySelector(${JSON.stringify(selector)})`), kind + ' result', 210000)
    if (kind === 'IMAGE') {
      uploadTimeToDisplayMs[kind] = Math.round(performance.now() - started)
      await click('Dual map views')
      const resumeUrl = new URL(url)
      resumeUrl.searchParams.set('verify-reload', '1')
      await call('Page.navigate', { url: resumeUrl.href })
      await waitFor(() => evaluate("location.search.includes('verify-reload=1') && !!document.querySelector('.sg-root')"), 'new dashboard document')
      await click('Live observation')
      await waitFor(() => evaluate("!!document.querySelector('img[alt=\"Annotated upload\"]')"), 'restored media result')
    }
    if (kind === 'VIDEO') {
      await waitFor(() => evaluate(`(() => {
        const video = document.querySelector('video.sg-media-output')
        if (video.error) throw new Error('Annotated video decoding failed')
        return video.readyState >= 2 && video.videoWidth > 0
      })()`), 'annotated video decode')
    }
    previews.push(kind)
    if (kind === 'VIDEO') uploadTimeToDisplayMs[kind] = Math.round(performance.now() - started)
  }
  process.stdout.write(JSON.stringify({ ...mapEvidence, previews, uploadTimeToDisplayMs, mediaResultRestoredAfterReload: true, webmDecoded: true }))
  }
  }
  }
} finally {
  socket?.close()
  edge.kill()
  await rm(profile, { recursive: true, force: true, maxRetries: 4 }).catch(() => {})
}
