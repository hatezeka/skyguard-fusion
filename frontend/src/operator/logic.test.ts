import {describe,it,expect} from 'vitest'
import {acceptedGeo,mapPoints,parseLive,stats,tacticalPoints,validateDetection,validCoordinate,normalizePath} from './logic'
import type {Dataset} from './types'
import {emptyDataset} from './types'
describe('SkyGuard operator scientific guards',()=>{
 it('rejects UNKNOWN identity',()=>expect(validateDetection({id:'d',frame_id:'f',session_id:'s',frame_index:0,captured_at:'2026-01-01T00:00:00Z',classification:'UNKNOWN',confidence:.7,x1:0,y1:0,x2:2,y2:2,track_id:'1'})).toBe(false))
 it('rejects invalid coordinates',()=>{expect(validCoordinate(98,22)).toBe(false);expect(validCoordinate(42.8,20.9)).toBe(true)})
 it('rejects unsupported coordinates',()=>expect(acceptedGeo({id:'g',detection_id:'d',session_id:'s',classification:'DRONE',method:'UNKNOWN',evidence_ref:'',calibration_ref:'',latitude:42,longitude:21,range_m:0,uncertainty_m:0})).toBe(false))
 it('removes track data for UNKNOWN live classification',()=>{const x=parseLive({detections:[{classification:'UNKNOWN',track_id:123,bbox:[1,2,3,4]}]});expect(x.detections[0].track_id).toBeNull();expect(x.tracks).toHaveLength(0)})
 it('provides no fabricated geo or tactical points',()=>{const d={...emptyDataset} as Dataset;expect(mapPoints(d,'s')).toHaveLength(0);expect(tacticalPoints(d,'s')).toHaveLength(0);expect(stats(d,'s').fps).toBeNull()})
 it('rejects unsafe endpoint',()=>expect(()=>normalizePath('https://example.com/api')).toThrow())
})
