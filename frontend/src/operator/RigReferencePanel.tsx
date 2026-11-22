import {useEffect,useState} from 'react'
import type {OperatorApi} from './api'

export type RigConfiguration = {
  id:string|null
  status:string
  reference:Record<string,unknown>|null
}

const numeric = [
  ['latitude','Latitude · WGS84 °',-90,90],
  ['longitude','Longitude · WGS84 °',-180,180],
  ['altitude_m','Antenna altitude · m',undefined,undefined],
  ['horizontal_accuracy_m','Horizontal accuracy · m',0,undefined],
  ['vertical_accuracy_m','Vertical accuracy · m',0,undefined],
  ['heading_deg','True-north heading · °',0,359.999999],
  ['heading_uncertainty_deg','Heading uncertainty · °',0,180],
  ['pitch_deg','Pitch · °',-90,90],
  ['roll_deg','Roll · °',-180,180],
  ['offset_x','Antenna offset right · m',undefined,undefined],
  ['offset_y','Antenna offset down · m',undefined,undefined],
  ['offset_z','Antenna offset forward · m',undefined,undefined],
] as const

export function RigReferencePanel({api,onSaved}:{api:OperatorApi;onSaved:(value:RigConfiguration)=>void}){
  const [saved,setSaved]=useState<RigConfiguration|null>(null)
  const [error,setError]=useState<string|null>(null)
  const [busy,setBusy]=useState(false)
  useEffect(()=>{
    const controller=new AbortController()
    api.get<RigConfiguration>('/live/rig-reference',controller.signal).then(result=>{
      setSaved(result)
      onSaved(result)
    }).catch(error=>{if(!controller.signal.aborted)setError(error.message)})
    return()=>controller.abort()
  },[api,onSaved])
  const ref=saved?.reference
  const value=(key:string)=>key.startsWith('offset_')
    ? (Array.isArray(ref?.antenna_offset_xyz_m)?Number(ref.antenna_offset_xyz_m['xyz'.indexOf(key.slice(-1))]):undefined)
    : typeof ref?.[key]==='number'?Number(ref[key]):undefined
  return <section aria-label="Geospatial reference / Camera rig location" className="sg-media-lab">
    <h2>Geospatial reference / Camera rig location</h2>
    <p role="status">{saved?.status??'LOADING'} {saved?.id&&<> · {saved.id}</>}</p>
    <p>Left optical center · +X right, +Y down, +Z forward. Enter the GPS antenna offset from that origin. Use true north and ellipsoidal height for projection.</p>
    {error&&<p role="alert">{error}</p>}
    <form key={saved?.id??saved?.status??'loading'} className="sg-media-cols" onSubmit={async event=>{
      event.preventDefault()
      setBusy(true);setError(null)
      try{
        const form=new FormData(event.currentTarget)
        const number=(key:string)=>{const raw=form.get(key);return typeof raw==='string'&&raw.trim()!==''?Number(raw):null}
        const date=(key:string)=>{
          const raw=String(form.get(key)??'').trim()
          if(!raw)return null
          if(!/(?:Z|[+-]\d{2}:\d{2})$/i.test(raw))throw new Error('Use an explicit time zone offset.')
          return new Date(raw).toISOString()
        }
        const payload:Record<string,unknown>=Object.fromEntries(numeric.filter(([key])=>!key.startsWith('offset_')).map(([key])=>[key,number(key)]))
        const offset=['offset_x','offset_y','offset_z'].map(number)
        if(offset.some(v=>v!==null)&&offset.some(v=>v===null))throw new Error('Enter all three antenna offsets.')
        Object.assign(payload,{antenna_offset_xyz_m:offset.every(v=>v!==null)?offset:null,
          source:form.get('source'),altitude_datum:form.get('altitude_datum')||null,
          captured_at:date('captured_at'),valid_until:date('valid_until'),evidence_ref:form.get('evidence_ref')||null})
        const result=await api.post<RigConfiguration>('/live/rig-reference',payload)
        setSaved(result);onSaved(result)
      }catch(error){setError(error instanceof Error?error.message:String(error))}
      finally{setBusy(false)}
    }}>
      {numeric.map(([key,label,min,max])=><label key={key}>{label}<input type="number" name={key} min={min} max={max} step="any"
        defaultValue={value(key)} required={key==='latitude'||key==='longitude'}/></label>)}
      <label>GPS source<select name="source" defaultValue={String(ref?.source??'manual')}>
        <option value="manual">Manual</option><option value="phone">Phone</option><option value="gnss">GNSS</option><option value="other">Other</option>
      </select></label>
      <label>Vertical datum<select name="altitude_datum" defaultValue={String(ref?.altitude_datum??'')}>
        <option value="">Unknown</option><option value="ellipsoidal">Ellipsoidal</option><option value="orthometric">Orthometric</option>
      </select></label>
      <label>Reference time · ISO 8601 with offset<input name="captured_at" required placeholder="2026-11-22T12:00:00Z" defaultValue={String(ref?.captured_at??'')}/></label>
      <label>Valid until · ISO 8601 with offset<input name="valid_until" defaultValue={String(ref?.valid_until??'')}/></label>
      <label>Measured reference evidence<input name="evidence_ref" maxLength={160} defaultValue={String(ref?.evidence_ref??'')}/></label>
      <button className="sg-btn primary" disabled={busy||!saved}>Save and validate reference</button>
    </form>
  </section>
}
