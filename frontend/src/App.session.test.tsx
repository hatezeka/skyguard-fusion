/* @vitest-environment jsdom */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import App from './App'
import { liveCameraApi, type LiveStatus } from './api/liveCamera'

vi.mock('./AirspaceMap', () => ({ default: () => null }))
vi.mock('./TrajectoryOverlay', () => ({ default: () => null }))
vi.mock('./CameraManager', () => ({ default: () => null }))
vi.mock('./useSynchronizedFrame', () => ({
  useSynchronizedFrame: () => ({ frame: null, error: null }),
}))
vi.mock('./api/liveCamera', () => ({
  liveCameraApi: {
    status: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    frame: vi.fn(),
    streamUrl: '/mock-stream',
  },
}))

function status(state: 'idle' | 'running' | 'stopped'): LiveStatus {
  return {
    state,
    frame_index: null,
    detections: [],
    tracks: [],
    trajectories: [],
    processing_seconds: null,
    error: null,
  } as LiveStatus
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: Error) => void
  const promise = new Promise<T>((r, j) => {
    resolve = r
    reject = j
  })
  return { promise, resolve, reject }
}

function cameraButton(container: HTMLElement, kind: 'start' | 'stop') {
  const button = container.querySelector<HTMLButtonElement>(
    `button.button-${kind}`,
  )
  if (!button) {
    throw new Error(`Camera ${kind} button selector not found`)
  }
  return button
}

describe('SkyGuard App session concurrency', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(liveCameraApi.status).mockResolvedValue(status('idle'))
    vi.mocked(liveCameraApi.start).mockResolvedValue(status('running'))
    vi.mocked(liveCameraApi.stop).mockResolvedValue(status('stopped'))
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('blocks overlapping start operations', async () => {
    const pending = deferred<LiveStatus>()
    vi.mocked(liveCameraApi.start).mockReturnValue(pending.promise)

    const { container } = render(<App />)

    await waitFor(() => {
      expect(liveCameraApi.status).toHaveBeenCalled()
    })

    const start = cameraButton(container, 'start')

    fireEvent.click(start)
    fireEvent.click(start)

    expect(liveCameraApi.start).toHaveBeenCalledTimes(1)

    await act(async () => {
      pending.resolve(status('running'))
      await pending.promise
    })

    await waitFor(() => {
      expect(liveCameraApi.status).toHaveBeenCalledTimes(2)
    })
  })

  it('blocks overlapping stop operations', async () => {
    vi.mocked(liveCameraApi.status)
      .mockResolvedValueOnce(status('running'))
      .mockResolvedValue(status('stopped'))

    const pending = deferred<LiveStatus>()
    vi.mocked(liveCameraApi.stop).mockReturnValue(pending.promise)

    const { container } = render(<App />)

    await waitFor(() => {
      expect(liveCameraApi.status).toHaveBeenCalled()
      expect(cameraButton(container, 'stop').disabled).toBe(false)
    })

    fireEvent.click(cameraButton(container, 'stop'))
    fireEvent.click(cameraButton(container, 'stop'))

    expect(liveCameraApi.stop).toHaveBeenCalledTimes(1)

    await act(async () => {
      pending.resolve(status('stopped'))
      await pending.promise
    })

    await waitFor(() => {
      expect(liveCameraApi.status).toHaveBeenCalledTimes(2)
    })
  })

  it('ignores a delayed status error from the old generation', async () => {
    const oldRequest = deferred<LiveStatus>()

    vi.mocked(liveCameraApi.status)
      .mockResolvedValueOnce(status('idle'))
      .mockImplementationOnce(() => oldRequest.promise)
      .mockResolvedValue(status('running'))

    const { container } = render(<App />)

    // The second request comes from the regular one-second poll.
    await waitFor(() => {
      expect(liveCameraApi.status).toHaveBeenCalledTimes(2)
    }, { timeout: 3500 })

    fireEvent.click(cameraButton(container, 'start'))

    await waitFor(() => {
      expect(liveCameraApi.start).toHaveBeenCalledTimes(1)
      expect(liveCameraApi.status).toHaveBeenCalledTimes(3)
    })

    await act(async () => {
      oldRequest.reject(new Error('STALE_SESSION_ERROR_MARKER'))
      await oldRequest.promise.catch(() => undefined)
    })

    expect(container.textContent).not.toContain(
      'STALE_SESSION_ERROR_MARKER',
    )
  })

  it('releases the start lock after an API failure', async () => {
    const pending = deferred<LiveStatus>()

    vi.mocked(liveCameraApi.start)
      .mockImplementationOnce(() => pending.promise)
      .mockResolvedValueOnce(status('running'))

    const { container } = render(<App />)

    await waitFor(() => {
      expect(container.textContent).toContain('API CONNECTED')
      expect(cameraButton(container, 'start').disabled).toBe(false)
    })

    fireEvent.click(cameraButton(container, 'start'))

    expect(liveCameraApi.start).toHaveBeenCalledTimes(1)

    await act(async () => {
      pending.reject(new Error('START_RETRY_TEST'))
      await pending.promise.catch(() => undefined)
    })

    await waitFor(() => {
      expect(cameraButton(container, 'start').disabled).toBe(false)
    })

    fireEvent.click(cameraButton(container, 'start'))

    await waitFor(() => {
      expect(liveCameraApi.start).toHaveBeenCalledTimes(2)
    })
  })
})
