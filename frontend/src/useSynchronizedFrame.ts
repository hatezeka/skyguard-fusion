import { useEffect, useState } from 'react'
import {
  liveCameraApi,
  type LiveSynchronizedFrame,
} from './api/liveCamera'

interface SynchronizedFrameState {
  frame: LiveSynchronizedFrame | null
  error: string | null
}

export function useSynchronizedFrame(
  enabled: boolean,
  sessionVersion: number,
): SynchronizedFrameState {
  const [result, setResult] = useState<
    SynchronizedFrameState & {
      version: number
      enabled: boolean
    }
  >({
    frame: null,
    error: null,
    version: sessionVersion,
    enabled,
  })

  useEffect(() => {
    if (!enabled) {
      return
    }

    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const poll = async () => {
      try {
        const frame = await liveCameraApi.frame()

        if (!cancelled) {
          setResult(previous => {
            if (
              previous.version === sessionVersion &&
              previous.enabled &&
              previous.frame !== null &&
              frame.frame_index <= previous.frame.frame_index
            ) {
              return previous
            }

            return {
              frame,
              error: null,
              version: sessionVersion,
              enabled: true,
            }
          })
        }
      } catch (error) {
        if (!cancelled) {
          setResult(previous => ({
            frame:
              previous.version === sessionVersion && previous.enabled
                ? previous.frame
                : null,
            error: error instanceof Error ? error.message : String(error),
            version: sessionVersion,
            enabled: true,
          }))
        }
      } finally {
        if (!cancelled) {
          timer = setTimeout(poll, 250)
        }
      }
    }

    void poll()

    return () => {
      cancelled = true
      if (timer !== undefined) {
        clearTimeout(timer)
      }
    }
  }, [enabled, sessionVersion])

  if (
    !enabled ||
    result.version !== sessionVersion ||
    !result.enabled
  ) {
    return { frame: null, error: null }
  }

  return { frame: result.frame, error: result.error }
}
