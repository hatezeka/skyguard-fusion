// @vitest-environment jsdom
import {cleanup,render,screen} from '@testing-library/react'
import {afterEach,expect,it} from 'vitest'
import {activeObjectCounts,checkedFrameCounts,ObjectCountPanel,type FrameResult} from './ObjectCountPanel'

afterEach(cleanup)

const frame:FrameResult={
  analysis_status:'COMPLETED',
  detections:[
    {classification:'DRONE',bbox:[1,2,15,16],track_id:null},
    {classification:'BIRD',bbox:[18,4,46,22],track_id:null}
  ],
  counts:{DRONE:1,BIRD:1,UNKNOWN:0},
  total_detected_objects:2
}

it('renders the verified object counts by public class',()=>{
  render(<ObjectCountPanel counts={checkedFrameCounts(frame)} heading="ANALYSIS COMPLETE"/>)
  expect(screen.getByText('DRONES').nextElementSibling?.textContent).toBe('1')
  expect(screen.getByText('BIRDS').nextElementSibling?.textContent).toBe('1')
  expect(screen.getByText('UNKNOWN').nextElementSibling?.textContent).toBe('0')
  expect(screen.getByText('TOTAL DETECTED OBJECTS').querySelector('strong')?.textContent).toBe('2')
})

it('withholds mismatched, malformed and missing results',()=>{
  expect(checkedFrameCounts({...frame,total_detected_objects:1})).toBeNull()
  expect(checkedFrameCounts({...frame,detections:[{classification:'DRONE',bbox:[3,3,2,2],track_id:null}]})).toBeNull()
  expect(checkedFrameCounts({})).toBeNull()
  render(<ObjectCountPanel counts={null} heading="Image analysis"/>)
  expect(screen.getByRole('status').textContent).toBe('INCOMPLETE')
})

it('counts a live track once and counts UNKNOWN only in the current frame',()=>{
  const box:[number,number,number,number]=[0,0,5,5]
  const drone={classification:'DRONE' as const,bbox:box,track_id:'17',confidence:.8,frame_index:4}
  const unknown={classification:'UNKNOWN' as const,bbox:box,track_id:null,confidence:null}
  expect(activeObjectCounts({
    detectionsComplete:true,tracks:[drone,drone],detections:[unknown]
  })).toEqual({DRONE:1,BIRD:0,UNKNOWN:1})
  expect(activeObjectCounts({
    detectionsComplete:true,tracks:[drone,{...drone,classification:'BIRD'}],detections:[]
  })).toBeNull()
})