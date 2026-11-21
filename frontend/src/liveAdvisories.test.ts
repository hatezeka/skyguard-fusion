import { describe, expect, it } from 'vitest'
import { alertScope, mergeAdvisories, parseAdvisory } from './api/liveAdvisories'
const SID='a'.repeat(32), CAM='cam_1', ID='c'.repeat(64)
const evt=(override:Record<string,unknown>={})=>({
  event_id:ID,track_id:4,frame_index:3,confidence:.95,session_id:SID,camera_id:CAM,
  event_type:'operator_review_advisory',severity:'attention',object_class_model_label:'drone',
  reason_codes:['MODEL_DRONE_LABEL'],ground_truth_verified:false,facility_breach_verified:false,
  target_gps:null,target_range_m:null,...override,
})
const sample=(override:Record<string,unknown>={})=>({
  camera_id:CAM,session_id:SID,frame_index:3,alerts:[evt()],
  evaluation_basis:'model_and_tracker_evidence_only',coordinate_system:'image_pixels',
  sampling_mode:'on_demand_latest_snapshot',sampled_frames_only:true,alert_archive_persistent:false,
  ground_truth_verified:false,target_gps_available:false,facility_distance_available:false,
  geofence_breach_verified:false,automated_response_allowed:false,...override,
})
describe('41.3 client advisory safety and identity',()=>{
  it('accepts valid sampled attention',()=>expect(parseAdvisory(sample(),CAM,SID).alerts[0].track_id).toBe(4))
  it('accepts empty alert batch',()=>expect(parseAdvisory(sample({alerts:[]}),CAM,SID).alerts).toEqual([]))
  it('rejects response from previous or unregistered camera/session',()=>{
    expect(()=>parseAdvisory(sample(),CAM,'b'.repeat(32))).toThrow()
    expect(()=>parseAdvisory(sample(),'../invalid',SID)).toThrow()
    expect(()=>parseAdvisory(sample({camera_id:'different'}),CAM,SID)).toThrow()
  })
  it('rejects physical claims and critical severity',()=>{
    for(const flag of ['target_gps_available','facility_distance_available','geofence_breach_verified','automated_response_allowed','ground_truth_verified']){
      expect(()=>parseAdvisory(sample({[flag]:true}),CAM,SID)).toThrow()
    }
    for(const update of [{severity:'critical'},{target_gps:{lat:42}},{target_range_m:100},{ground_truth_verified:true},{facility_breach_verified:true}]){
      expect(()=>parseAdvisory(sample({alerts:[evt(update)]}),CAM,SID)).toThrow()
    }
  })
  it('rejects unsafe event IDs, duplicate events and oversized arrays',()=>{
    expect(()=>parseAdvisory(sample({alerts:[evt({event_id:'bad'})]}),CAM,SID)).toThrow()
    expect(()=>parseAdvisory(sample({alerts:[evt(),evt()]}),CAM,SID)).toThrow()
    expect(()=>parseAdvisory(sample({alerts:Array(513).fill(evt())}),CAM,SID)).toThrow()
  })
  it('rejects future frame events and non-finite confidences',()=>{
    expect(()=>parseAdvisory(sample({alerts:[evt({frame_index:4})]}),CAM,SID)).toThrow()
    expect(()=>parseAdvisory(sample({alerts:[evt({confidence:Infinity})]}),CAM,SID)).toThrow()
  })
  it('rejects fake continuous frame monitoring or persistent archiving',()=>{
    expect(()=>parseAdvisory(sample({sampled_frames_only:false}),CAM,SID)).toThrow()
    expect(()=>parseAdvisory(sample({alert_archive_persistent:true}),CAM,SID)).toThrow()
  })
  it('deduplicates replays and refuses older frames',()=>{
    const batch=parseAdvisory(sample(),CAM,SID)
    const first=mergeAdvisories(null,batch)
    expect(mergeAdvisories(first,batch).events).toHaveLength(1)
    expect(mergeAdvisories(first,{...batch,frame_index:1,alerts:[]}).frame).toBe(3)
  })
  it('resets history on new session',()=>{
    const first=mergeAdvisories(null,parseAdvisory(sample(),CAM,SID))
    const next='b'.repeat(32)
    const result=mergeAdvisories(first,parseAdvisory(sample({session_id:next,frame_index:1,alerts:[]}),CAM,next))
    expect(result.events).toHaveLength(0)
    expect(result.key).not.toBe(first.key)
  })
  it('validates scope identity',()=>{
    expect(alertScope(CAM,SID)).toBe(`${CAM}:${SID}`)
    expect(()=>alertScope(CAM,'nope')).toThrow()
  })
})
