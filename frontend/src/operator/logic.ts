import type {Camera, Dataset, Detection, Geo, LiveSnapshot, ObjClass, Track} from './types'
export const validCoordinate = (lat: unknown, lon: unknown):lat is number => typeof lat==='number' && Number.isFinite(lat) && lat>=-90 && lat<=90 && typeof lon==='number' && Number.isFinite(lon) && lon>=-180 && lon<=180
export const safeDate = (value:string|null|undefined) => {const t=value?Date.parse(value):NaN;return Number.isFinite(t)?new Date(t).toLocaleString('sq-AL',{hour12:false}):'—'}
export const shortId=(s:string|null|undefined)=>s? (s.length>16?`${s.slice(0,8)}…${s.slice(-5)}`:s):'—'
export const num=(n:number|null|undefined,d=1)=>typeof n==='number' && Number.isFinite(n)?n.toFixed(d):'—'
export const score=(n:number|null|undefined)=>typeof n==='number'&&n>=0&&n<=1?`${(n*100).toFixed(1)}%`:'—'
export const validClass=(v:unknown):v is ObjClass=>v==='DRONE'||v==='BIRD'||v==='UNKNOWN'
export function validateTrack(t:Track){return t.classification==='DRONE'||t.classification==='BIRD'}
export function validateDetection(d:Detection){return validClass(d.classification)&& (d.classification!=='UNKNOWN'||d.track_id===null)}
export function acceptedGeo(g:Geo){return (g.classification==='DRONE'||g.classification==='BIRD') && !!g.calibration_ref && !!g.evidence_ref && typeof g.uncertainty_m==='number' && Number.isFinite(g.uncertainty_m)&&g.uncertainty_m>=0 && (validCoordinate(g.latitude,g.longitude)||typeof g.range_m==='number'&&Number.isFinite(g.range_m)&&g.range_m>=0)}
export function mapPoints(data:Dataset,sessionId:string){
  const det=new Map(data.detections.filter(d=>d.session_id===sessionId&&validateDetection(d)).map(d=>[d.id,d]));
  return data.geo.filter(g=>g.session_id===sessionId&&acceptedGeo(g)&&validCoordinate(g.latitude,g.longitude)).flatMap(g=>{
    const d=det.get(g.detection_id);if(!d||d.classification==='UNKNOWN'||!d.track_id)return [];
    return [{...g,lat:g.latitude as number,lon:g.longitude as number,track_id:d.track_id,captured_at:d.captured_at}]
  })
}
export function tacticalPoints(data:Dataset, sessionId:string, camera?:Camera){
  if(!camera||!camera.calibration_ref||!validCoordinate(camera.latitude,camera.longitude)) return []
  const lat0=camera.latitude as number,lon0=camera.longitude as number
  return mapPoints(data,sessionId).map(p=>{
    const north=(p.lat-lat0)*111132.92
    const east=(p.lon-lon0)*111320*Math.cos(lat0*Math.PI/180)
    return {...p,east,north,distance:Math.hypot(north,east)}
  }).filter(p=>Number.isFinite(p.east)&&Number.isFinite(p.north)&&p.distance<=200000)
}
export function sessionDetections(data:Dataset,id:string){return data.detections.filter(d=>d.session_id===id&&validateDetection(d))}
export function selectLatestByTrack(ds:Detection[]){const map=new Map<string,Detection>();for(const d of ds){if(!d.track_id||d.classification==='UNKNOWN')continue;const old=map.get(d.track_id);if(!old||Date.parse(old.captured_at)<Date.parse(d.captured_at))map.set(d.track_id,d)}return map}
export function stats(data:Dataset,sessionId:string){const ds=sessionDetections(data,sessionId), tracks=data.tracks.filter(t=>t.session_id===sessionId&&validateTrack(t)), alerts=data.alerts.filter(a=>a.session_id===sessionId), p=data.performance.filter(x=>x.session_id===sessionId).sort((a,b)=>Date.parse(b.measured_at)-Date.parse(a.measured_at));return {drones:ds.filter(d=>d.classification==='DRONE').length,birds:ds.filter(d=>d.classification==='BIRD').length,unknown:ds.filter(d=>d.classification==='UNKNOWN').length,tracks:tracks.length,alerts:alerts.filter(a=>a.status==='OPEN').length,fps:p.find(x=>x.fps!==null)?.fps??null,latency:p.find(x=>x.latency_ms!==null)?.latency_ms??null}}
export function parseLive(raw:unknown):LiveSnapshot{
  if(typeof raw!=='object'||!raw)throw new Error('Invalid live response')
  const r=raw as Record<string,unknown>;const objects=Array.isArray(r.detections)?r.detections:[]
  const detections=objects.slice(0,500).flatMap(o=>{
    if(!o||typeof o!=='object')return []
    const d=o as Record<string,unknown>,cls=String(d.classification??d.class??'').toUpperCase()
    if(!validClass(cls)||!Array.isArray(d.bbox)||d.bbox.length!==4||!d.bbox.every(x=>typeof x==='number'&&Number.isFinite(x)&&x>=0)||!(d.bbox[2]>d.bbox[0]&&d.bbox[3]>d.bbox[1]))return []
    const tid=(typeof d.track_id==='string'||typeof d.track_id==='number')?String(d.track_id):null
    const confidence=typeof d.confidence==='number'&&Number.isFinite(d.confidence)&&d.confidence>=0&&d.confidence<=1?d.confidence:null
    return [{classification:cls,confidence,bbox:d.bbox as [number,number,number,number],track_id:cls==='UNKNOWN'?null:tid}]
  })
  const tracks=detections.filter(x=>x.track_id&&x.classification!=='UNKNOWN').map(x=>({track_id:x.track_id as string,classification:x.classification as 'DRONE'|'BIRD',confidence:x.confidence,bbox:x.bbox,frame_index:typeof r.frame_index==='number'?r.frame_index:0}))
  const metric=(v:unknown)=>typeof v==='number'&&Number.isFinite(v)&&v>=0?v:null
  return {state:typeof r.state==='string'?r.state:undefined,frame_index:metric(r.frame_index),captured_at:typeof r.captured_at==='string'&&!Number.isNaN(Date.parse(r.captured_at))?r.captured_at:null,fps:metric(r.fps),latency_ms:metric(r.latency_ms),detections,tracks,source:typeof r.source==='string'?r.source:'SKYGUARD LIVE',image_mime:r.image_mime==='image/jpeg'?'image/jpeg':null,image_base64:r.image_mime==='image/jpeg'&&typeof r.image_base64==='string'&&r.image_base64.length<=12000000&&/^[A-Za-z0-9+/]+={0,2}$/.test(r.image_base64)?r.image_base64:null,recorder:typeof r.recorder==='object'&&r.recorder?r.recorder as LiveSnapshot['recorder']:undefined}
}
export function exportFile(filename:string,content:string,mime='application/json'){const blob=new Blob([content],{type:mime});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000)}
export function normalizePath(s:string){if(!s.startsWith('/api/')||s.includes('..')||s.includes('?')||s.includes('#')||s.includes('\\')||s.length>200)throw new Error('Use a local /api/... endpoint');return s}
