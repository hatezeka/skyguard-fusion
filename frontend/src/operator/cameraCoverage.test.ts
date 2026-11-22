import {describe,expect,it} from 'vitest'
import {offsetCoordinate,orientationSector,scaleRings,verifiedCameraCoverage} from './cameraCoverage'
import type {Benchmark, Camera} from './types'

const camera:Camera={id:'fixture-camera',name:'Fixture',source_kind:'USB',latitude:42,longitude:21,
 calibration_ref:'measured-fixture',created_at:'2026-01-01T00:00:00Z'}
const profile={
 camera_id:camera.id,calibration_ref:camera.calibration_ref,latitude:camera.latitude,longitude:camera.longitude,
 evidence_ref:'fixture-report',fixed_camera:true,heading_deg:90,horizontal_fov_deg:45,
 trials:[{latitude:42,longitude:21.001,horizontal_m:100,bearing_deg:90,passed:true}],
 verified_area:[[21,42],[21.001,42],[21.001,42.001]],
}
const benchmark:Benchmark={id:'fixture',validation_level:'PHYSICAL_VERIFIED',metrics:{camera_coverage_v1:profile}}

describe('Measured camera coverage',()=>{
 it('computes horizontal markers and adjusts ring distances with zoom',()=>{
  expect(offsetCoordinate([0,0],90,1000)[0]).toBeCloseTo(.008993,5)
  const far=scaleRings([21,42],11)
  const near=scaleRings([21,42],15)
  expect(far[0].meters).toBeGreaterThan(near[0].meters)
  expect(far[0].line).toHaveLength(73)
  expect(orientationSector([21,42],90,45,100)[0]).toEqual([21,42])
 })
 it('shows one-direction results without painting an area',()=>{
  const result=verifiedCameraCoverage([benchmark],camera)
  expect(result?.maximumVerified).toBe(100)
  expect(result?.orientation).toEqual({heading:90,fieldOfView:45})
  expect(result?.area).toEqual([])
 })
 it('rejects unverified tests and evidence for a changed position',()=>{
  expect(verifiedCameraCoverage([{...benchmark,validation_level:'SOFTWARE_ONLY'}],camera)).toBeNull()
  expect(verifiedCameraCoverage([benchmark],{...camera,latitude:41})).toBeNull()
 })
 it('accepts only an explicitly supplied polygon backed by more than one measured direction',()=>{
  const record={...profile,trials:[...profile.trials,{latitude:42.001,longitude:21,horizontal_m:110,bearing_deg:0,passed:true}]}
  const result=verifiedCameraCoverage([{...benchmark,metrics:{camera_coverage_v1:record}}],camera)
  expect(result?.area).toEqual(profile.verified_area)
  expect(result?.maximumVerified).toBe(110)
 })
})
