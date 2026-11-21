import type {Alert, Audit, Benchmark, Camera, Dataset, Detection, Evidence, Geo, Performance, Session, Summary, Track} from './types'
import {emptyDataset} from './types'
import {normalizePath} from './logic'

/** Local development only: direct service bearer key must never be shipped to public browser clients. */
export class OperatorApi {
  private readonly token: () => string | null

  constructor(token: () => string | null) {
    this.token = token
  }
  private async request<T>(path:string,init?:RequestInit):Promise<T>{
    const deployed=import.meta.env.VITE_SKYGUARD_DEPLOYED==='true';const key=this.token();if(!deployed&&!key)throw new Error('Local operator key is required; never store a service key in production frontend')
    const res=await fetch(`/api/persistence${path}`,{...init,headers:{...(deployed?{}:{'Authorization':`Bearer ${key}`}),...(init?.body?{'Content-Type':'application/json'}:{}),...init?.headers},credentials:'same-origin',cache:'no-store',signal:init?.signal})
    if(!res.ok){if(res.status===401||res.status===403)throw new Error('Authentication rejected (401/403)');if(res.status===503)throw new Error('Persistence API disabled (503): configure SKYGUARD_DB_API_KEY');throw new Error(`Persistence API error HTTP ${res.status}`)}
    return await res.json() as T
  }
  get<T>(p:string,signal?:AbortSignal){return this.request<T>(p,{signal})}
  post<T>(p:string,body:unknown){return this.request<T>(p,{method:'POST',body:JSON.stringify(body)})}
  health(){return this.get<{status:string;backend:string}>('/health')}
  startUsbCamera(index:number,fusion=false){return this.post<unknown>('/live/start',{camera_index:index,source_type:'usb',source_location:index,fusion_enabled:fusion})}
  stopUsbCamera(){return this.post<unknown>('/live/stop',{})}
  cameras(){return this.get<Camera[]>('/cameras')}
  sessions(){return this.get<Session[]>('/sessions')}
  detections(sid:string){return this.get<Detection[]>(`/detections?session_id=${encodeURIComponent(sid)}&limit=200`)}
  tracks(sid:string){return this.get<Track[]>(`/tracks?session_id=${encodeURIComponent(sid)}&limit=200`)}
  geo(sid:string){return this.get<Geo[]>(`/geospatial?session_id=${encodeURIComponent(sid)}&limit=200`)}
  alerts(sid:string){return this.get<Alert[]>(`/alerts?session_id=${encodeURIComponent(sid)}&limit=200`)}
  performance(sid:string){return this.get<Performance[]>(`/performance?session_id=${encodeURIComponent(sid)}&limit=200`)}
  summary(sid:string){return this.get<Summary>(`/sessions/${encodeURIComponent(sid)}/summary`)}
  evidence(sid:string){return this.get<Evidence[]>(`/evidence?session_id=${encodeURIComponent(sid)}&limit=200`)}
  benchmarks(){return this.get<Benchmark[]>('/benchmarks?limit=100')}
  audit(){return this.get<Audit[]>('/audit?limit=100')}
  history(tid:string){return this.get<Array<{observation:{id:string;observed_at:string;frame_index:number};detection:Detection}>>(`/tracks/${encodeURIComponent(tid)}/history?limit=200`)}
  acknowledge(id:string,actor:string){return this.post<Alert>(`/alerts/${encodeURIComponent(id)}/acknowledge`,{actor})}
  addCamera(body:{name:string;source_kind:string;calibration_ref:string|null;latitude:number|null;longitude:number|null}){return this.post<Camera>('/cameras',body)}
  async load(sessionId:string):Promise<Dataset>{
    const [cameras,sessions,benchmarks,audit]=await Promise.all([this.cameras(),this.sessions(),this.benchmarks(),this.audit()])
    if(!sessionId)return {...emptyDataset,cameras,sessions,benchmarks,audit}
    const [detections,tracks,geo,alerts,performance,evidence,summary]=await Promise.all([this.detections(sessionId),this.tracks(sessionId),this.geo(sessionId),this.alerts(sessionId),this.performance(sessionId),this.evidence(sessionId),this.summary(sessionId)])
    return {cameras,sessions,benchmarks,audit,detections,tracks,geo,alerts,performance,evidence,summary}
  }
}
export async function fetchLiveStatus(route:string,signal?:AbortSignal,token?:string){
  const path=normalizePath(route)
  if(!path.startsWith('/api/persistence/live/'))throw new Error('Only the authenticated SkyGuard live bridge is supported')
  if(!token)throw new Error('Local persistence key is required for live status')
  const r=await fetch(path,{cache:'no-store',credentials:'same-origin',headers:{Authorization:`Bearer ${token}`},signal})
  if(!r.ok)throw new Error(`Live status HTTP ${r.status}`)
  const length=r.headers.get('content-length');if(length&&Number(length)>12000000)throw new Error('Live response too large')
  return await r.json() as unknown
}
