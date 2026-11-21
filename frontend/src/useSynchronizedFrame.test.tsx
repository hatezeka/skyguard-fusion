/* @vitest-environment jsdom */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useSynchronizedFrame } from './useSynchronizedFrame'
import { liveCameraApi } from './api/liveCamera'

vi.mock('./api/liveCamera', () => ({
  liveCameraApi: {
    frame: vi.fn(),
  },
}))

type Frame = Awaited<ReturnType<typeof liveCameraApi.frame>>

function makeFrame(index: number): Frame {
  return {
    frame_index: index,
  } as Frame
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(r => {
    resolve = r
  })
  return { promise, resolve }
}

describe('SkyGuard synchronized frame isolation', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('does not replace a newer frame with an older index', async () => {
    vi.useFakeTimers()

    vi.mocked(liveCameraApi.frame)
      .mockResolvedValueOnce(makeFrame(5))
      .mockResolvedValueOnce(makeFrame(4))

    const { result } = renderHook(() =>
      useSynchronizedFrame(true, 1),
    )

    await act(async () => {
      await Promise.resolve()
    })

    expect(result.current.frame?.frame_index).toBe(5)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250)
    })

    expect(result.current.frame?.frame_index).toBe(5)
    expect(liveCameraApi.frame).toHaveBeenCalledTimes(2)
  })

  it('rejects a delayed response from an older session', async () => {
    const oldRequest = deferred<Frame>()

    vi.mocked(liveCameraApi.frame)
      .mockImplementationOnce(() => oldRequest.promise)
      .mockResolvedValueOnce(makeFrame(1))

    const { result, rerender } = renderHook(
      ({ version }) => useSynchronizedFrame(true, version),
      { initialProps: { version: 1 } },
    )

    rerender({ version: 2 })

    await act(async () => {
      await Promise.resolve()
    })

    expect(result.current.frame?.frame_index).toBe(1)

    await act(async () => {
      oldRequest.resolve(makeFrame(99))
      await oldRequest.promise
    })

    expect(result.current.frame?.frame_index).toBe(1)
    expect(liveCameraApi.frame).toHaveBeenCalledTimes(2)
  })

  it('does not request frames while disabled', async () => {
    vi.mocked(liveCameraApi.frame)
      .mockResolvedValue(makeFrame(3))

    const { result, rerender } = renderHook(
      ({ enabled }) => useSynchronizedFrame(enabled, 1),
      { initialProps: { enabled: false } },
    )

    expect(result.current.frame).toBeNull()
    expect(liveCameraApi.frame).not.toHaveBeenCalled()

    rerender({ enabled: true })

    await act(async () => {
      await Promise.resolve()
    })

    expect(result.current.frame?.frame_index).toBe(3)

    rerender({ enabled: false })

    expect(result.current.frame).toBeNull()
  })
})