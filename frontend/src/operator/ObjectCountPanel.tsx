import type {LiveSnapshot,ObjClass} from './types'

const categories = [['DRONE','DRONES'],['BIRD','BIRDS'],['UNKNOWN','UNKNOWN']] as const
export type ObjectCounts = Record<ObjClass, number>
export type FrameResult = {
  analysis_status?: string
  detections?: {classification:string|null;bbox:number[];track_id:string|null}[]
  counts?: Record<string,number>
  total_detected_objects?: number
}

export function checkedFrameCounts(frame:FrameResult):ObjectCounts|null {
  if(frame.analysis_status!=='COMPLETED'||!Array.isArray(frame.detections)) return null
  const counts:ObjectCounts={DRONE:0,BIRD:0,UNKNOWN:0}
  for(const detection of frame.detections){
    const key=detection?.classification
    const box=detection?.bbox
    if((key!=='DRONE'&&key!=='BIRD'&&key!=='UNKNOWN')||!Array.isArray(box)||box.length!==4||
      !box.every(value=>Number.isFinite(value)&&value>=0)||box[2]<=box[0]||box[3]<=box[1]) return null
    counts[key]++
  }
  if(frame.total_detected_objects!==frame.detections.length||categories.some(([key])=>frame.counts?.[key]!==counts[key])) return null
  return counts
}

export function activeObjectCounts(live:Pick<LiveSnapshot,'detectionsComplete'|'tracks'|'detections'>):ObjectCounts|null {
  if (!live.detectionsComplete) return null
  const byId = new Map<string,'DRONE'|'BIRD'>()
  for (const track of live.tracks) {
    const previous = byId.get(track.track_id)
    if (previous && previous !== track.classification) return null
    byId.set(track.track_id, track.classification)
  }
  const counts:ObjectCounts = {DRONE:0,BIRD:0,UNKNOWN:live.detections.filter(d=>d.classification==='UNKNOWN').length}
  for (const label of byId.values()) counts[label]++
  return counts
}

export function ObjectCountPanel({counts,heading,unavailable='INCOMPLETE'}:{counts:ObjectCounts|null;heading:string;unavailable?:string}){
  return <section aria-label={heading} className="sg-object-counts">
    <h3>{heading}</h3>
    {counts?<><dl>{categories.map(([key,label])=><div key={key} data-class={key}><dt>{label}</dt><dd>{counts[key]}</dd></div>)}</dl>
      <p>TOTAL DETECTED OBJECTS <strong>{counts.DRONE+counts.BIRD+counts.UNKNOWN}</strong></p></>:<p role="status">{unavailable}</p>}
  </section>
}
