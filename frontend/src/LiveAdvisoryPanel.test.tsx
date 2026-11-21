// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest'
import { liveCameraApi, type LiveStatus } from './api/liveCamera'
import LiveAdvisoryPanel from './LiveAdvisoryPanel'
vi.mock('./api/liveCamera',()=>({liveCameraApi:{evaluateLatestAdvisories:vi.fn()}}))
const SID='a'.repeat(32)
const status=(p:Record<string,unknown>={}):LiveStatus=>({
  state:'running',camera_id:'cam_1',session_id:SID,frame_index:3,
  detections:[],tracks:[],processing_seconds:null,error:null,...p,
} as LiveStatus)
const event=()=>({
  event_id:'c'.repeat(64),track_id:4,frame_index:3,confidence:.95,session_id:SID,camera_id:'cam_1',
  event_type:'operator_review_advisory',severity:'attention',object_class_model_label:'drone',
  reason_codes:['MODEL_DRONE_LABEL'],ground_truth_verified:false,facility_breach_verified:false,
  target_gps:null,target_range_m:null,
})
const response=(p:Record<string,unknown>={})=>({
  camera_id:'cam_1',session_id:SID,frame_index:3,alerts:[event()],
  evaluation_basis:'model_and_tracker_evidence_only',coordinate_system:'image_pixels',
  sampling_mode:'on_demand_latest_snapshot',sampled_frames_only:true,alert_archive_persistent:false,
  ground_truth_verified:false,target_gps_available:false,facility_distance_available:false,
  geofence_breach_verified:false,automated_response_allowed:false,...p,
})
beforeEach(()=>vi.clearAllMocks())
afterEach(()=>cleanup())
describe('Phase 41.3 live advisory panel',()=>{
  it('does not poll a disconnected API',()=>{
    render(<LiveAdvisoryPanel liveStatus={status()} connected={false}/>)
    expect(screen.getByText(/API e shkëputur/)).toBeTruthy()
    expect(liveCameraApi.evaluateLatestAdvisories).not.toHaveBeenCalled()
  })
  it('does not poll when camera is unregistered',()=>{
    render(<LiveAdvisoryPanel liveStatus={status({camera_id:null})} connected/>)
    expect(screen.getByText(/Kërkohet kamerë e regjistruar/)).toBeTruthy()
    expect(liveCameraApi.evaluateLatestAdvisories).not.toHaveBeenCalled()
  })
  it('shows only operator-review model evidence, not physical confirmation',async()=>{
    vi.mocked(liveCameraApi.evaluateLatestAdvisories).mockResolvedValue(response())
    render(<LiveAdvisoryPanel liveStatus={status()} connected/>)
    await waitFor(()=>expect(screen.getByText(/Shqyrtim operatori · ID 4/)).toBeTruthy())
    expect(screen.getByText(/jo dronë të konfirmuar fizikisht/)).toBeTruthy()
    expect(screen.getByText(/Kjo nuk vërteton identitet fizik/)).toBeTruthy()
  })
  it('rejects a spoofed GPS or range flag',async()=>{
    vi.mocked(liveCameraApi.evaluateLatestAdvisories).mockResolvedValue(response({target_gps_available:true}))
    render(<LiveAdvisoryPanel liveStatus={status()} connected/>)
    await waitFor(()=>expect(screen.getByRole('alert')).toBeTruthy())
    expect(screen.queryByText(/Shqyrtim operatori · ID 4/)).toBeNull()
  })
  it('rejects other session data',async()=>{
    vi.mocked(liveCameraApi.evaluateLatestAdvisories).mockResolvedValue(response({session_id:'b'.repeat(32)}))
    render(<LiveAdvisoryPanel liveStatus={status()} connected/>)
    await waitFor(()=>expect(screen.getByRole('alert')).toBeTruthy())
    expect(screen.queryByText(/Shqyrtim operatori · ID 4/)).toBeNull()
  })
  it('hides old session events immediately on restart',async()=>{
    vi.mocked(liveCameraApi.evaluateLatestAdvisories).mockResolvedValue(response())
    const {rerender}=render(<LiveAdvisoryPanel liveStatus={status()} connected/>)
    await waitFor(()=>expect(screen.getByText(/Shqyrtim operatori · ID 4/)).toBeTruthy())
    rerender(<LiveAdvisoryPanel liveStatus={status({session_id:'b'.repeat(32)})} connected/>)
    expect(screen.queryByText(/Shqyrtim operatori · ID 4/)).toBeNull()
  })
  it('shows safe no-alert state',async()=>{
    vi.mocked(liveCameraApi.evaluateLatestAdvisories).mockResolvedValue(response({alerts:[]}))
    render(<LiveAdvisoryPanel liveStatus={status()} connected/>)
    await waitFor(()=>expect(screen.getByText(/Nuk ka sinjalizime në frame-et/)).toBeTruthy())
  })
})
