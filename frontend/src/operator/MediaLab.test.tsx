// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import MediaLab from './MediaLab'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

test.each([
  { fileName: 'photo.jpg', mime: 'image/jpeg', kind: 'IMAGE' },
  { fileName: 'clip.mp4', mime: 'video/mp4', kind: 'VIDEO' },
])('selecting $fileName starts analysis and renders its result', async ({ fileName, mime, kind }) => {
  const prediction = '/api/media/jobs/test-job/prediction'
  const request = vi.fn()
    .mockResolvedValueOnce(Response.json({ job_id: 'test-job' }, { status: 202 }))
    .mockResolvedValueOnce(Response.json({
      id: 'test-job',
      kind,
      status: 'DONE',
      frames_processed: 1,
      session_id: null,
      error: null,
      summary: {
        conflict_review_total: 1,
        prediction_url: prediction,
        preview_jpeg_base64: kind === 'IMAGE' ? 'fixture-preview' : undefined,
      },
    }))
  vi.stubGlobal('fetch', request)
  const file = new File(['input'], fileName, { type: mime })

  render(<MediaLab />)
  fireEvent.change(screen.getByLabelText('Choose image or video'), { target: { files: [file] } })

  expect(await screen.findByRole('heading', { name: 'Job: DONE' })).toBeTruthy()
  expect(request).toHaveBeenNthCalledWith(1, '/api/media/upload', expect.objectContaining({
    method: 'POST',
    headers: expect.objectContaining({ 'X-Skyguard-Media-Type': kind }),
    body: file,
  }))
  expect(request).toHaveBeenNthCalledWith(2, '/api/media/jobs/test-job', expect.anything())
  expect(screen.getByText('Conflicting proposals for review: 1')).toBeTruthy()
  if (kind === 'IMAGE') {
    expect(screen.getByRole('img', { name: 'Annotated upload' })).toBeTruthy()
  } else {
    expect(screen.getByRole('link', { name: 'Open annotated video' }).getAttribute('href')).toBe(prediction)
  }
})

test('returning to MediaLab resumes the accepted job', async () => {
  const request = vi.fn().mockResolvedValue(Response.json({
    id: 'existing-job', kind: 'IMAGE', status: 'DONE', frames_processed: 1,
    session_id: null, error: null, summary: { preview_jpeg_base64: 'fixture-preview' },
  }))
  vi.stubGlobal('fetch', request)

  render(<MediaLab activeJobId="existing-job" />)

  expect(await screen.findByRole('img', { name: 'Annotated upload' }, { timeout: 3000 })).toBeTruthy()
  expect(request).toHaveBeenCalledWith('/api/media/jobs/existing-job', expect.anything())
})

test('an unavailable stored job is cleared without repeatedly requesting it', async () => {
  const request = vi.fn().mockResolvedValue(Response.json({ detail: 'Unknown job ID' }, { status: 404 }))
  const onJobAccepted = vi.fn()
  vi.stubGlobal('fetch', request)

  render(<MediaLab activeJobId="missing-job" onJobAccepted={onJobAccepted} />)

  expect(await screen.findByText(/previous media job is unavailable/, {}, { timeout: 3000 })).toBeTruthy()
  expect(onJobAccepted).toHaveBeenCalledWith()
  expect(request).toHaveBeenCalledTimes(1)
})

test('the IP camera view reports a lost stream after waiting for the first frame', async () => {
  const request = vi.fn()
    .mockResolvedValueOnce(Response.json({ state: 'running' }))
    .mockResolvedValueOnce(Response.json({ state: 'running', image_base64: null }))
    .mockResolvedValueOnce(Response.json({ state: 'error', image_base64: null }))
  vi.stubGlobal('fetch', request)
  render(<MediaLab persistenceKey="test-key" />)
  fireEvent.change(screen.getByLabelText('Private network RTSP URL'), { target: { value: 'rtsp://127.0.0.1/test' } })
  fireEvent.click(screen.getByRole('button', { name: 'Start IP camera' }))
  expect(await screen.findByText('Waiting for the first IP camera frame.')).toBeTruthy()
  expect(await screen.findByText(/IP camera signal unavailable/, {}, { timeout: 3500 })).toBeTruthy()
  expect(screen.queryByRole('img', { name: 'Annotated IP camera frame' })).toBeNull()
})

test('MediaLab renders confirmed targets and excludes UNKNOWN trajectories', async () => {
  const points = [{ frame: 0, x_px: 10, y_px: 20 }, { frame: 1, x_px: 15, y_px: 25 }]
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({
    id: 'trajectory-test', kind: 'VIDEO', status: 'DONE', frames_processed: 2,
    session_id: null, error: null, summary: { track_paths_px: [
      { track_id: 'verified-drone', classification: 'DRONE', points },
      { track_id: 'invalid-unknown-track', classification: 'UNKNOWN', points },
    ] },
  })))
  render(<MediaLab activeJobId="trajectory-test" />)
  expect(await screen.findByText('Recorded pixel trajectories (1)', {}, { timeout: 3000 })).toBeTruthy()
  expect(screen.getByText('verified-drone')).toBeTruthy()
  expect(screen.queryByText('invalid-unknown-track')).toBeNull()
})
