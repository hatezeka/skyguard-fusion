import { spawn } from 'node:child_process'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
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
  await call('Network.enable')
  await call('Page.navigate', { url })
  async function connectPersistence() {
    await click('Integration settings')
    const key = process.env.SKYGUARD_DB_API_KEY
    if (!key) throw new Error('Missing verification key')
    await evaluate(`(() => {
      const input = document.querySelector('input[type=password]')
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, ${JSON.stringify(key)})
      input.dispatchEvent(new Event('input', { bubbles: true }))
    })()`)
    await click('Connect persistence')
    await waitFor(() => evaluate("document.body.innerText.includes('DATABASE CONNECTED')"), 'database connection')
  }
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
    process.stdout.write(JSON.stringify({ beforeReload, afterReload }))
  } else {
  const viewportEvidence = []
  if (mode === '--responsive') {
    const screens = [['phone', 360, 800], ['tablet', 768, 1024], ['desktop', 1440, 900]]
    const pages = ['Command center','Live observation','Object tracking','Dual map views','Alert management',
      'Evidence ledger','Performance','Camera registry','Scientific validation','Integration settings']
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
} finally {
  socket?.close()
  edge.kill()
  await rm(profile, { recursive: true, force: true, maxRetries: 4 }).catch(() => {})
}
