import {describe,it,expect} from 'vitest'
import {acceptedGeo,cameraPixelPaths,droneGeoPaths,mapPoints,parseLive,stats,tacticalPoints,validateDetection,validCoordinate,normalizePath} from './logic'
import type {Dataset,Detection} from './types'
import {emptyDataset} from './types'
describe('SkyGuard operator scientific guards',()=>{
 it('rejects UNKNOWN identity',()=>expect(validateDetection({id:'d',frame_id:'f',session_id:'s',frame_index:0,captured_at:'2026-01-01T00:00:00Z',classification:'UNKNOWN',confidence:.7,x1:0,y1:0,x2:2,y2:2,track_id:'1'})).toBe(false))
 it('rejects invalid coordinates',()=>{expect(validCoordinate(98,22)).toBe(false);expect(validCoordinate(42.8,20.9)).toBe(true)})
 it('rejects unsupported coordinates',()=>expect(acceptedGeo({id:'g',detection_id:'d',session_id:'s',classification:'DRONE',method:'UNKNOWN',evidence_ref:'',calibration_ref:'',latitude:42,longitude:21,range_m:0,uncertainty_m:0})).toBe(false))
 it('removes track data for UNKNOWN live classification',()=>{const x=parseLive({detections:[{classification:'UNKNOWN',track_id:123,bbox:[1,2,3,4]}]});expect(x.detections[0].track_id).toBeNull();expect(x.tracks).toHaveLength(0)})
 it('provides no fabricated geo or tactical points',()=>{const d={...emptyDataset} as Dataset;expect(mapPoints(d,'s')).toHaveLength(0);expect(tacticalPoints(d,'s')).toHaveLength(0);expect(stats(d,'s').fps).toBeNull()})
 it('rejects unsafe endpoint',()=>expect(()=>normalizePath('https://example.com/api')).toThrow())
})

describe('Dual map trajectory evidence',()=>{
 const first:Detection={id:'d1',frame_id:'f1',session_id:'s',frame_index:1,captured_at:'2026-01-01T00:00:01Z',
  classification:'DRONE',confidence:.85,x1:20,y1:30,x2:30,y2:50,track_id:'database-track'}
 const second={...first,id:'d2',frame_id:'f2',frame_index:2,captured_at:'2026-01-01T00:00:02Z',x1:30,x2:40}
 const unknown:Detection={...first,id:'unverified',frame_index:3,classification:'UNKNOWN'}
 const data:Dataset={...emptyDataset,detections:[second,unknown,first],tracks:[
  {id:'database-track',session_id:'s',external_track_id:'observed-7',classification:'DRONE',
   first_seen:first.captured_at,last_seen:second.captured_at},
 ]}
 it('uses ordered bbox centers and the Object tracking ID, excluding UNKNOWN',()=>{
  expect(cameraPixelPaths(data,'s')).toEqual([{id:'database-track',label:'observed-7',classification:'DRONE',
   points:[{frame:1,x:25,y:40},{frame:2,x:35,y:40}]}])
  expect(mapPoints(data,'s')).toEqual([])
 })
 it('only puts evidence-backed drone samples on the geographic path',()=>{
  const base={session_id:'s',classification:'DRONE' as const,method:'REGISTERED_GEO',
   calibration_ref:'calibration',evidence_ref:'evidence',range_m:null,uncertainty_m:1}
  const positioned:Dataset={...data,geo:[
   {...base,id:'geo-2',detection_id:'d2',latitude:42.1,longitude:21.1},
   {...base,id:'invalid',detection_id:'unverified',latitude:43,longitude:22},
   {...base,id:'geo-1',detection_id:'d1',latitude:42,longitude:21},
  ]}
  expect(droneGeoPaths(mapPoints(positioned,'s'))[0].points.map(p=>[p.lon,p.lat])).toEqual([[21,42],[21.1,42.1]])
  expect(cameraPixelPaths(positioned,'s')).toEqual([])
 })
})
