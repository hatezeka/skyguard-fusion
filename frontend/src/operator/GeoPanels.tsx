import {useEffect, useMemo, useRef, useState} from 'react'
import {Map as MapLibreMap, Marker, Popup, NavigationControl, ScaleControl, LngLatBounds, setWorkerUrl} from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import type {Camera,Dataset} from './types'
import {mapPoints,tacticalPoints,shortId,num,validCoordinate} from './logic'
import 'maplibre-gl/dist/maplibre-gl.css'
import './real-map.css'

// Browser-side MapLibre worker: the Vite build bundles it locally, with no CDN code dependency.
setWorkerUrl(workerUrl)

const BASE_STYLES={
 dark:'https://tiles.openfreemap.org/styles/dark',
 street:'https://tiles.openfreemap.org/styles/liberty',
} as const

type BaseStyle='dark'|'street'|'satellite'
type Props={data:Dataset;sessionId:string;camera?:Camera;selected:string|null;onSelect:(v:string)=>void}

// The initial center is a neutral map VIEW, never evidence of the camera or any object's GPS.
const INITIAL_VIEW:[number,number]=[20.9,42.7]
const mapTilerKey=(import.meta.env.VITE_MAPTILER_API_KEY||'').trim()
const styleUrl=(kind:BaseStyle):string=>kind==='satellite' && mapTilerKey
 ? `https://api.maptiler.com/maps/hybrid/style.json?key=${encodeURIComponent(mapTilerKey)}`
 : BASE_STYLES[kind==='street'?'street':'dark']

function infoNode(title:string,lines:readonly string[]):HTMLDivElement {
 const wrapper=document.createElement('div')
 wrapper.className='sg-map-popup'
 const heading=document.createElement('strong')
 heading.textContent=title
 wrapper.appendChild(heading)
 for(const line of lines){const div=document.createElement('div');div.textContent=line;wrapper.appendChild(div)}
 return wrapper
}

export function GeoPanel({data,sessionId,camera,selected,onSelect}:Props){
 const points=useMemo(()=>mapPoints(data,sessionId),[data,sessionId])
 const knownCam=!!camera&&!!camera.calibration_ref&&validCoordinate(camera.latitude,camera.longitude)
 const nodeRef=useRef<HTMLDivElement|null>(null)
 const mapRef=useRef<MapLibreMap|null>(null)
 const markersRef=useRef<Marker[]>([])
 const fittedRef=useRef<string|null>(null)
 const appliedStyleRef=useRef<BaseStyle>(mapTilerKey?'satellite':'dark')
 const [style,setStyle]=useState<BaseStyle>(mapTilerKey?'satellite':'dark')
 const [ready,setReady]=useState(false)
 const [error,setError]=useState<string|null>(null)

 useEffect(()=>{
  const node=nodeRef.current
  if(!node)return
  let alive=true
  const map=new MapLibreMap({
   container:node,
   style:styleUrl(mapTilerKey?'satellite':'dark'),
   center:INITIAL_VIEW,
   zoom:6,
   pitch:mapTilerKey?45:0,
   bearing:0,
   attributionControl:{},
   maxPitch:55,
  })
  mapRef.current=map
  map.addControl(new NavigationControl({showCompass:true,visualizePitch:true}),'top-right')
  map.addControl(new ScaleControl({maxWidth:120,unit:'metric'}),'bottom-left')
  map.on('load',()=>{if(alive){setReady(true);setError(null);map.resize()}})
  map.on('error',event=>{
   if(!alive)return
   const message=event.error?.message||'Map source could not be loaded'
   setError(message)
  })
  const observer=typeof ResizeObserver!=='undefined'?new ResizeObserver(()=>map.resize()):null
  observer?.observe(node)
  return ()=>{
   alive=false
   observer?.disconnect()
   markersRef.current.forEach(marker=>marker.remove())
   markersRef.current=[]
   mapRef.current=null
   map.remove()
  }
 },[])

 useEffect(()=>{
  const map=mapRef.current
  if(!map||style===appliedStyleRef.current)return
  appliedStyleRef.current=style
  setError(null)
  map.setStyle(styleUrl(style))
  map.easeTo({pitch:style==='satellite'?45:0,duration:400})
 },[style])

 useEffect(()=>{
  const map=mapRef.current
  if(!map||!ready)return
  markersRef.current.forEach(marker=>marker.remove())
  markersRef.current=[]
  const bounds=new LngLatBounds()
  const signature:string[]=[]
  if(knownCam&&camera){
   const coordinate:[number,number]=[camera.longitude!,camera.latitude!]
   bounds.extend(coordinate)
   signature.push(`CAM:${camera.id}:${coordinate.join(',')}`)
   const element=document.createElement('button')
   element.type='button';element.className='sg-map-marker sg-map-sensor'
   element.setAttribute('aria-label',`Camera ${camera.name}`)
   element.title=`Referenced camera: ${camera.name}`
   element.textContent='⌖'
   const marker=new Marker({element,anchor:'center'})
    .setLngLat(coordinate)
    .setPopup(new Popup({offset:14}).setDOMContent(infoNode(`CAMERA · ${camera.name}`,[`WGS84: ${num(camera.latitude,6)}, ${num(camera.longitude,6)}`,`Calibration: ${camera.calibration_ref}`])))
    .addTo(map)
   markersRef.current.push(marker)
  }
  for(const point of points){
   const coordinate:[number,number]=[point.lon,point.lat]
   bounds.extend(coordinate)
   signature.push(`${point.id}:${coordinate.join(',')}`)
   const element=document.createElement('button')
   element.type='button'
   element.className=`sg-map-marker ${point.classification==='DRONE'?'sg-map-drone':'sg-map-bird'}${selected===point.track_id?' sg-map-selected':''}`
   element.textContent=point.classification==='DRONE'?'D':'B'
   element.title=`${point.classification} / ${point.track_id}`
   element.setAttribute('aria-label',`${point.classification} track ${point.track_id}`)
   element.addEventListener('click',()=>onSelect(point.track_id))
   const marker=new Marker({element,anchor:'center'})
    .setLngLat(coordinate)
    .setPopup(new Popup({offset:16}).setDOMContent(infoNode(`${point.classification} · ${shortId(point.track_id)}`,[`WGS84: ${num(point.lat,6)}, ${num(point.lon,6)}`,`Uncertainty: ±${num(point.uncertainty_m)} m`,`Method: ${point.method}`,`Evidence: ${point.evidence_ref}`])))
    .addTo(map)
   markersRef.current.push(marker)
  }
  const fitKey=signature.slice().sort().join('|')
  // Fit ONLY real evidence-backed georeferenced positions, never pixel-based guesses.
  if(fitKey&&fitKey!==fittedRef.current){
   if(signature.length===1){
    const midpoint=bounds.getCenter()
    map.easeTo({center:midpoint,zoom:12,duration:500})
   }else map.fitBounds(bounds,{padding:58,maxZoom:14,duration:500})
   fittedRef.current=fitKey
  }
 },[points,knownCam,camera,selected,onSelect,ready])

 return <div className="geo-view sg-real-map-view">
  <div className="panel-head compact"><div><h3>Real Map · WGS84</h3><p>OpenFreeMap / OpenStreetMap · mapped positions require evidence</p></div><span className="subbadge">{points.length} verified object fixes</span></div>
  <div className="sg-map-toolbar" role="group" aria-label="Geographic base map styles">
   <button type="button" className={style==='dark'?'active':''} onClick={()=>setStyle('dark')}>Dark</button>
   <button type="button" className={style==='street'?'active':''} onClick={()=>setStyle('street')}>Streets</button>
   <button type="button" className={style==='satellite'?'active':''} disabled={!mapTilerKey} title={!mapTilerKey?'Set VITE_MAPTILER_API_KEY for licensed MapTiler imagery':undefined} onClick={()=>setStyle('satellite')}>Satellite {!mapTilerKey?'(key required)':''}</button>
  </div>
  <div className="geo-leaflet sg-maplibre-host" ref={nodeRef} role="region" aria-label="Interactive real geographic map"/>
  {error&&<div role="status" className="sg-map-status-error">Map tiles unavailable: {error}. Check the network, map provider and browser WebGL. Do not treat the map as operational until tiles load.</div>}
  {!knownCam&&points.length===0&&<div className="sg-map-status-info">Basemap displayed for orientation only. No verified camera GPS or target positions; no markers are fabricated.</div>}
  {points.length>0&&<div className="sg-map-status-info">Verified markers only · Camera/DRONE/BIRD evidence · UNKNOWN never mapped</div>}
  <div className="geo-key"><span><i className="keydot" style={{background:'#f97357'}}/>DRONE</span><span><i className="keydot" style={{background:'#41c4b7'}}/>BIRD</span><span><i className="keydot" style={{background:'#43baff'}}/>SENSOR (calibrated)</span></div>
 </div>
}
export function TacticalPanel({data,sessionId,camera,selected,onSelect}:Props){
 const p=useMemo(()=>tacticalPoints(data,sessionId,camera),[data,sessionId,camera]);const max=Math.max(100,...p.map(v=>v.distance+v.uncertainty_m));
 const px=(m:number)=>150+Math.max(-1,Math.min(1,m/max))*116;
 return <div className="tactical-view"><div className="panel-head compact"><div><h3>Tactical · sensor-relative</h3><p>North-up; measured position from registered geo only</p></div><span className="subbadge">{p.length} positioned</span></div>
 <div className="tactical-canvas"><svg role="img" aria-label="Sensor-relative tactical positions" viewBox="0 0 300 300" preserveAspectRatio="xMidYMid meet" className="tactical-svg">
 <defs><radialGradient id="radarFade"><stop offset="0%" stopColor="#164c66" stopOpacity="0.4"/><stop offset="100%" stopColor="#091628" stopOpacity="0"/></radialGradient></defs>
 <circle cx="150" cy="150" r="140" fill="url(#radarFade)"/>
 {[40,78,116].map(r=><circle key={r} cx="150" cy="150" r={r} stroke="#38536a" strokeOpacity=".45" fill="none" strokeDasharray="3 6"/>)}
 <path d="M150 8V292 M8 150H292" stroke="#45647b" strokeOpacity=".5" strokeDasharray="3 5"/><text x="150" y="19" fill="#819cb5" textAnchor="middle" fontSize="11">N</text><text x="285" y="145" fill="#819cb5" textAnchor="end" fontSize="11">E</text>
 {p.map(v=>{const x=px(v.east),y=px(-v.north),u=Math.max(2,116*v.uncertainty_m/max);return <g key={v.id} className="tactical-target" onClick={()=>onSelect(v.track_id)} tabIndex={0} role="button" aria-label={`Select ${v.track_id}`} onKeyDown={e=>{if(e.key==='Enter')onSelect(v.track_id)}}>
 <circle cx={x} cy={y} r={Math.min(u,90)} fill={v.classification==='DRONE'?'#f97357':'#41c4b7'} fillOpacity=".09" stroke={v.classification==='DRONE'?'#f97357':'#41c4b7'} strokeOpacity=".4"/>
 <circle cx={x} cy={y} r={v.track_id===selected?7:5} fill={v.classification==='DRONE'?'#ff916b':'#62d2c5'} stroke="#081526" strokeWidth="2"/>
 <text x={x+9} y={y-7} fill="#d8e9fb" fontSize="9">{shortId(v.track_id)}</text>
 </g>})}
 <circle cx="150" cy="150" r="6" fill="#43baff" stroke="white" strokeWidth="1.5"/><path d="M150 138l-4 9h8z" fill="#43baff"/>
 </svg>
 {!p.length&&<div className="tactical-empty">No validated relative coordinates<br/><small>Camera georeference + object evidence required</small></div>}
 <span className="scale-mark">Scale radius {Math.round(max).toLocaleString()} m</span></div>
 <div className="geo-key"><span><i className="keydot" style={{background:'#f97357'}}/>DRONE</span><span><i className="keydot" style={{background:'#41c4b7'}}/>BIRD</span><span><i className="keydot" style={{background:'#43baff'}}/>SENSOR</span></div>
 </div>
}
