import type {RigConfiguration} from './RigReferencePanel'
import {useEffect, useMemo, useRef, useState} from 'react'
import {Map as MapLibreMap, Marker, Popup, NavigationControl, ScaleControl, LngLatBounds, setWorkerUrl, type GeoJSONSource} from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import type {Camera,Dataset,Detection,LiveSnapshot} from './types'
import {acceptedGeo,cameraPixelPaths,geographicTrackPaths,mapPoints,tacticalPoints,shortId,num,validCoordinate} from './logic'
import 'maplibre-gl/dist/maplibre-gl.css'
import './real-map.css'
import {offsetCoordinate,orientationSector,scaleRings,verifiedCameraCoverage,type Coordinate} from './cameraCoverage'

// Browser-side MapLibre worker: the Vite build bundles it locally, with no CDN code dependency.
setWorkerUrl(workerUrl)

const BASE_STYLES={
 dark:'https://tiles.openfreemap.org/styles/dark',
 street:'https://tiles.openfreemap.org/styles/liberty',
} as const

type BaseStyle='dark'|'street'|'satellite'
type Props={rigReference?:RigConfiguration|null;data:Dataset;sessionId:string;camera?:Camera;selected:string|null;onSelect:(v:string)=>void;trackHistory?:Detection[];live?:LiveSnapshot|null}

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

export function GeoPanel({data,sessionId,camera:sessionCamera,selected,onSelect,rigReference}:Props){
 const [cameraChoice,setCameraChoice]=useState<{sessionId:string;cameraId:string}|null>(null)
 const camera=cameraChoice?.sessionId===sessionId
  ? data.cameras.find(item=>item.id===cameraChoice.cameraId)??sessionCamera : sessionCamera
 const [showCoverage,setShowCoverage]=useState(true)
 const coverage=useMemo(()=>camera?verifiedCameraCoverage(data.benchmarks,camera):null,[camera,data.benchmarks])
 const points=useMemo(()=>camera?.id===sessionCamera?.id?mapPoints(data,sessionId):[],[data,sessionId,camera?.id,sessionCamera?.id])
 const paths=useMemo(()=>geographicTrackPaths(points),[points])
 const droneLocations=new Set(paths.filter(path=>path.points[0]?.classification==='DRONE').map(path=>path.id))
 const missingDroneLocation=!droneLocations.size||data.tracks.some(track=>
  track.session_id===sessionId&&track.classification==='DRONE'&&!droneLocations.has(track.id))
 const trackLabels=useMemo(()=>new Map(data.tracks.filter(track=>track.session_id===sessionId)
  .map(track=>[track.id,track.external_track_id])),[data.tracks,sessionId])
 const knownCam=!!camera&&camera.source_kind!=='SIMULATED'&&!!camera.calibration_ref?.trim()&&validCoordinate(camera.latitude,camera.longitude)
 const [cameraProvenance,setCameraProvenance]=useState<{cameraId:string;latitude:number;longitude:number;source:string;uncertainty:number|null}|null>(null)
 useEffect(()=>{
  if(!camera||!knownCam)return
  const controller=new AbortController()
  const liveRef=camera.external_ref?.match(/^live:[a-z]+:([a-zA-Z0-9_-]+):[0-9a-f]+$/)?.[1]
  const external=liveRef??camera.external_ref
  const profileId=external&&external!=='unregistered'&&/^[a-zA-Z0-9_-]+$/.test(external)?external:camera.id
  void fetch('/api/v1/geo/cameras/'+encodeURIComponent(profileId),{signal:controller.signal,credentials:'same-origin'})
   .then(response=>response.ok?response.json():null)
   .then(profile=>{
    if(controller.signal.aborted||!profile||profile.camera_id!==profileId||
       profile.latitude!==camera.latitude||profile.longitude!==camera.longitude||
       !['gnss','manual'].includes(profile.position_source))return
    const accuracy=profile.position_accuracy_m
    setCameraProvenance({cameraId:camera.id,latitude:profile.latitude,longitude:profile.longitude,
     source:profile.position_source==='gnss'?'GNSS':'Matje manuale e regjistruar',
     uncertainty:typeof accuracy==='number'&&Number.isFinite(accuracy)&&accuracy>=0?accuracy:null})
   }).catch(()=>{})
  return()=>{controller.abort()}
 },[camera,knownCam])
 const provenance=camera&&cameraProvenance?.cameraId===camera.id&&
   cameraProvenance.latitude===camera.latitude&&cameraProvenance.longitude===camera.longitude?cameraProvenance:null
 const positionSource=coverage?.source??provenance?.source??'Profili i ruajtur i kamerës'
 const positionUncertainty=coverage?.uncertainty??provenance?.uncertainty??null
 const nodeRef=useRef<HTMLDivElement|null>(null)
 const mapRef=useRef<MapLibreMap|null>(null)
 const markersRef=useRef<Marker[]>([])
 const fittedRef=useRef<string|null>(null)
 const centeredRef=useRef<string|null>(null)
 const appliedStyleRef=useRef<BaseStyle>(mapTilerKey?'satellite':'dark')
 const [style,setStyle]=useState<BaseStyle>(mapTilerKey?'satellite':'dark')
 const [ready,setReady]=useState(false)
 useEffect(()=>{
  const map=mapRef.current
  const reference=rigReference?.reference
  if(!ready||!map||rigReference?.status!=='GEO_REFERENCE_VALID'||!reference)return
  const lat=reference.latitude,lon=reference.longitude
  if(typeof lat!=='number'||typeof lon!=='number'||!validCoordinate(lat,lon))return
  const element=document.createElement('div')
  element.className='sg-map-sensor'
  element.title='Rig GPS antenna'
  element.textContent='Rig GPS'
  const marker=new Marker({element}).setLngLat([lon,lat]).addTo(map)
  map.jumpTo({center:[lon,lat],zoom:Math.max(12,map.getZoom())})
  return()=>{marker.remove()}
 },[ready,rigReference])
 const [error,setError]=useState<string|null>(null)

 useEffect(()=>{
  const node=nodeRef.current
  if(!node)return
  let alive=true
  let map:MapLibreMap
  try{map=new MapLibreMap({
   container:node,
   style:styleUrl(mapTilerKey?'satellite':'dark'),
   center:INITIAL_VIEW,
   zoom:6,
   pitch:mapTilerKey?45:0,
   bearing:0,
   attributionControl:{},
   maxPitch:55,
  })}catch{setError("Harta gjeografike nuk mund të inicializohet në këtë shfletues.");return}
  mapRef.current=map
  map.addControl(new NavigationControl({showCompass:true,visualizePitch:true}),'top-right')
  map.addControl(new ScaleControl({maxWidth:120,unit:'metric'}),'bottom-left')
  let styleLoaded=false
  map.on('style.load',()=>{if(alive){styleLoaded=true;setReady(true);map.resize()}})
  const fallback=window.setTimeout(()=>{
   if(!alive||styleLoaded)return
   setError('Pllakat e hartës nuk u ngarkuan. Shtresat lokale vazhdojnë në sfond neutral.')
   map.setStyle({version:8,sources:{},layers:[{id:'sg-offline-background',type:'background',paint:{'background-color':'#10212e'}}]})
  },8000)
  map.on('error',event=>{
   if(!alive)return
   const message=event.error?.message||'Map source could not be loaded'
   setError(message)
  })
  const observer=typeof ResizeObserver!=='undefined'?new ResizeObserver(()=>map.resize()):null
  observer?.observe(node)
  return ()=>{
   alive=false
   window.clearTimeout(fallback)
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
  setReady(false)
  map.setStyle(styleUrl(style))
  map.easeTo({pitch:style==='satellite'?45:0,duration:400})
 },[style])

 useEffect(()=>{
  const map=mapRef.current
  if(!map)return
  const sourceId='sg-verified-object-history'
  const lines:GeoJSON.Feature[]=paths.filter(path=>path.points.length>1).map(path=>({
    type:'Feature' as const,
    properties:{selected:path.id===selected,classification:path.points[0].classification},
    geometry:{type:'LineString' as const,coordinates:path.points.map(point=>[point.lon,point.lat])},
   }))
  const uncertainty:GeoJSON.Feature[]=paths.flatMap(path=>{
   const latest=path.points.at(-1)
   if(!latest||latest.uncertainty_m<=0)return []
   const center:Coordinate=[latest.lon,latest.lat]
   return [{type:'Feature',properties:{classification:latest.classification},
    geometry:{type:'Polygon',coordinates:[Array.from({length:73},(_,i)=>offsetCoordinate(center,i*5,latest.uncertainty_m))]}}]
  })
  const collection:GeoJSON.FeatureCollection={type:'FeatureCollection',features:[...lines,...uncertainty]}
  const update=()=>{
   const source=map.getSource<GeoJSONSource>(sourceId)
   if(source){source.setData(collection);return}
   map.addSource(sourceId,{type:'geojson',data:collection})
   map.addLayer({
    id:sourceId,type:'line',source:sourceId,filter:['==',['geometry-type'],'LineString'],
    paint:{'line-color':['match',['get','classification'],'DRONE','#ff916b','#62d2c5'],'line-width':['case',['get','selected'],4,2],'line-dasharray':[2,1]},
   })
   map.addLayer({id:sourceId+'-uncertainty',type:'fill',source:sourceId,
    filter:['==',['geometry-type'],'Polygon'],paint:{'fill-color':['match',['get','classification'],'DRONE','#ff916b','#62d2c5'],'fill-opacity':.15}})
  }
  if(map.getSource(sourceId)||map.isStyleLoaded())update()
  map.on('style.load',update)
  return()=>{map.off('style.load',update)}
 },[paths,selected,ready])

 useEffect(()=>{
  const map=mapRef.current
  if(!map)return
  const sourceId='sg-camera-coverage'
  let labels:Marker[]=[]
  const coord:Coordinate|null=knownCam&&camera?[camera.longitude!,camera.latitude!]:null
  const refresh=(styleReady=false)=>{
   const source=map.getSource<GeoJSONSource>(sourceId)
   if(!source&&!styleReady&&!map.isStyleLoaded())return
   labels.forEach(label=>label.remove())
   labels=[]
   const features:GeoJSON.Feature[]=[]
   if(coord&&showCoverage){
    const rings=scaleRings(coord,map.getZoom())
    for(const ring of rings){
     features.push({type:'Feature',properties:{kind:'ring'},geometry:{type:'LineString',coordinates:ring.line}})
     const element=document.createElement('span')
     element.className='sg-map-ring-label'
     element.textContent=ring.meters.toLocaleString('sq-AL')+' m'
     labels.push(new Marker({element,anchor:'left'}).setLngLat(ring.label).addTo(map))
    }
    if(coverage?.orientation){
     const radius=rings.at(-1)!.meters
     features.push({type:'Feature',properties:{kind:'fov'},geometry:{type:'LineString',
      coordinates:orientationSector(coord,coverage.orientation.heading,coverage.orientation.fieldOfView,radius)}})
    }
    if(coverage?.area.length){
     const polygon=[...coverage.area,coverage.area[0]]
     features.push({type:'Feature',properties:{kind:'area'},geometry:{type:'Polygon',coordinates:[polygon]}})
    }
    for(const trial of coverage?.trials??[]){
     features.push({type:'Feature',properties:{kind:trial.passed?'passed':'failed'},
      geometry:{type:'Point',coordinates:trial.coordinate}})
    }
   }
   const data:GeoJSON.FeatureCollection={type:'FeatureCollection',features}
   if(source){source.setData(data);return}
   map.addSource(sourceId,{type:'geojson',data})
   map.addLayer({id:sourceId+'-area',type:'fill',source:sourceId,filter:['==',['get','kind'],'area'],
    paint:{'fill-color':'#38bba4','fill-opacity':.16}})
   map.addLayer({id:sourceId+'-rings',type:'line',source:sourceId,filter:['==',['get','kind'],'ring'],
    paint:{'line-color':'#72c7ed','line-opacity':.66,'line-width':1}})
   map.addLayer({id:sourceId+'-fov',type:'line',source:sourceId,filter:['==',['get','kind'],'fov'],
    paint:{'line-color':'#ffd186','line-dasharray':[3,2],'line-width':2}})
   map.addLayer({id:sourceId+'-trials',type:'circle',source:sourceId,
    filter:['match',['get','kind'],['passed','failed'],true,false],
    paint:{'circle-color':['match',['get','kind'],'passed','#38bba4','#e98b78'],'circle-radius':5,
     'circle-stroke-color':'#0b1a2a','circle-stroke-width':1}})
  }
  const onStyleLoad=()=>refresh(true)
  const onZoom=()=>refresh()
  refresh(ready)
  map.on('style.load',onStyleLoad)
  map.on('zoomend',onZoom)
  return ()=>{
   map.off('style.load',onStyleLoad)
   map.off('zoomend',onZoom)
   labels.forEach(label=>label.remove())
  }
 },[camera,knownCam,coverage,showCoverage,ready])

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
   element.setAttribute('aria-label',`Kamera ${camera.name}`)
   element.title=`Referenced camera: ${camera.name}`
   element.textContent='Kamera'
   const marker=new Marker({element,anchor:'center'})
    .setLngLat(coordinate)
    .setPopup(new Popup({offset:14}).setDOMContent(infoNode(camera.name,[
     `WGS84: ${num(camera.latitude,6)}, ${num(camera.longitude,6)}`,
     `Burimi: ${positionSource}`,
     `Referenca: ${camera.calibration_ref}`,
     ...(positionUncertainty!=null?[`Pasiguria e lokacionit: ±${num(positionUncertainty)} m`]:[]),
    ])))
    .addTo(map)
   markersRef.current.push(marker)
  }
  const latestPositions=new Set(paths.map(path=>path.points.at(-1)?.id))
  for(const point of points){
   const coordinate:[number,number]=[point.lon,point.lat]
   bounds.extend(coordinate)
   signature.push(`${point.id}:${coordinate.join(',')}`)
   if(!latestPositions.has(point.id))continue
   const label=trackLabels.get(point.track_id)??point.track_id
   const element=document.createElement('button')
   element.type='button'
   element.className=`sg-map-marker ${point.classification==='DRONE'?'sg-map-drone':'sg-map-bird'}${selected===point.track_id?' sg-map-selected':''}`
   element.textContent=point.classification==='DRONE'?'D':'B'
   element.title=`${point.classification} / ${label}`
   element.setAttribute('aria-label',`${point.classification} track ${label}`)
   element.addEventListener('click',()=>onSelect(point.track_id))
   const marker=new Marker({element,anchor:'center'})
    .setLngLat(coordinate)
    .setPopup(new Popup({offset:16}).setDOMContent(infoNode(`${point.classification} · ${shortId(label)}`,[`WGS84: ${num(point.lat,6)}, ${num(point.lon,6)}`,`Uncertainty: ±${num(point.uncertainty_m)} m`,`Method: ${point.method}`,`Evidence: ${point.evidence_ref}`])))
    .addTo(map)
   markersRef.current.push(marker)
  }
  const fitKey=signature.slice().sort().join('|')
  // Fit ONLY real evidence-backed georeferenced positions, never pixel-based guesses.
  const cameraKey=knownCam&&camera?JSON.stringify([camera.id,camera.longitude,camera.latitude]):null
  if(cameraKey&&centeredRef.current!==cameraKey&&camera){
   map.jumpTo({center:[camera.longitude!,camera.latitude!],zoom:Math.max(12,map.getZoom())})
   centeredRef.current=cameraKey
   fittedRef.current=fitKey
  }else if(!cameraKey&&fitKey&&fitKey!==fittedRef.current){
   if(signature.length===1){
    const midpoint=bounds.getCenter()
    map.easeTo({center:midpoint,zoom:12,duration:500})
   }else map.fitBounds(bounds,{padding:58,maxZoom:14,duration:500})
   fittedRef.current=fitKey
  }
  if(!cameraKey)centeredRef.current=null
 },[points,paths,knownCam,camera,selected,onSelect,ready,trackLabels,positionSource,positionUncertainty])

 return <div className="geo-view sg-real-map-view">
  <div className="panel-head compact"><div><h3>Real Map · WGS84</h3><p>OpenFreeMap / OpenStreetMap · mapped positions require evidence</p></div><span className="subbadge">{points.length} verified object fixes</span></div>
  <div className="sg-map-coverage-controls">
   <label>Kamera
    <select aria-label="Kamera në hartë" value={camera?.id??''}
     onChange={event=>setCameraChoice({sessionId,cameraId:event.target.value})}>
     <option value="">Zgjidh kamerën</option>
     {data.cameras.map(item=><option key={item.id} value={item.id}>{item.name}</option>)}
    </select>
   </label>
   <label><input type="checkbox" checked={showCoverage} onChange={event=>setShowCoverage(event.target.checked)}/>
    Mbulueshmëria e kamerës
   </label>
  </div>
  <div className="sg-map-toolbar" role="group" aria-label="Geographic base map styles">
   <button type="button" className={style==='dark'?'active':''} onClick={()=>setStyle('dark')}>Dark</button>
   <button type="button" className={style==='street'?'active':''} onClick={()=>setStyle('street')}>Streets</button>
   <button type="button" className={style==='satellite'?'active':''} disabled={!mapTilerKey} title={!mapTilerKey?'Set VITE_MAPTILER_API_KEY for licensed MapTiler imagery':undefined} onClick={()=>setStyle('satellite')}>Satellite {!mapTilerKey?'(key required)':''}</button>
  </div>
  <div className="geo-leaflet sg-maplibre-host" ref={nodeRef} role="region" aria-label="Interactive real geographic map"/>
  {error&&<div role="status" className="sg-map-status-error">Map tiles unavailable: {error}. Check the network, map provider and browser WebGL. Do not treat the map as operational until tiles load.</div>}
  {missingDroneLocation&&<div role="status" className="sg-map-status-info">Vendndodhja e dronit e padisponueshme</div>}
  {!knownCam&&<div role="status" className="sg-map-status-info">Lokacioni i kamerës i pakonfirmuar</div>}
  {knownCam&&camera&&<p className="sg-map-status-info">{camera.name} · Burimi: {positionSource}
   {positionUncertainty!=null&&` · pasiguri ±${num(positionUncertainty)} m`}</p>}
  {showCoverage&&knownCam&&<div className="sg-map-status-info">
   Unazat tregojnë distancë horizontale nga kamera.
   {coverage?.orientation&&' Vija e ndërprerë tregon fushëpamjen orientuese.'}
   {coverage?.maximumVerified!=null?<div>
    {coverage.multipleDirections
     ?`Distanca më e largët e kaluar në pikat e testuara: ${num(coverage.maximumVerified,0)} m`
     :`Detektimi u verifikua deri në ${num(coverage.maximumVerified,0)} m në drejtimin e testuar`}
    <div>Prova: {coverage.evidenceRef}</div>
   </div>:<div>Mbulueshmëria e testuar e padisponueshme</div>}
   {!!coverage?.trials.length&&<ul>{coverage.trials.map((trial,index)=><li key={index}>
    {num(trial.horizontalMeters,0)} m · {num(trial.bearing,0)}° · {trial.passed?'Kaloi':'Dështoi'}
   </li>)}</ul>}
  </div>}
  {!knownCam&&points.length===0&&<div className="sg-map-status-info">Basemap displayed for orientation only. No verified camera GPS or target positions; no markers are fabricated.</div>}
  {points.length>0&&<div className="sg-map-status-info">Verified markers only · Camera/DRONE/BIRD evidence · UNKNOWN never mapped</div>}
  <div className="geo-key"><span><i className="keydot" style={{background:'#f97357'}}/>DRONE</span><span><i className="keydot" style={{background:'#41c4b7'}}/>BIRD</span><span><i className="keydot" style={{background:'#43baff'}}/>SENSOR (calibrated)</span></div>
 </div>
}
function AngularRadar({live, data, sessionId, selected, onSelect}:Pick<Props,'live'|'data'|'sessionId'|'selected'|'onSelect'>){
 const frame=live?.state==='running'&&live.persistence_session_id===sessionId?live:null
 const tracked=new Map(data.tracks.filter(t=>t.session_id===sessionId).map(t=>[t.external_track_id,t]))
 const anglePoints=(frame?.detections??[]).flatMap(d=>{
  if((d.classification!=='DRONE'&&d.classification!=='BIRD')||!d.track_id||d.relative_bearing_deg==null)return []
  const track=tracked.get(d.track_id)
  if(!track||track.classification!==d.classification)return []
  return [{id:track.id,label:track.external_track_id,classification:track.classification,bearing:d.relative_bearing_deg}]
 })
 const point=(a:number,r:number)=>{
  const t=a*Math.PI/180
  return [200+r*Math.sin(t),208-r*Math.cos(t)]
 }
 const pathFor=(from:number,to:number,r:number)=>Array.from({length:25},(_,i)=>point(from+(to-from)*i/24,r).join(',')).join(' ')
 const fov=frame?.geometry?.horizontal_fov_deg
 return <section className="sg-angular-radar">
  <div className="panel-head compact"><div><h3>Tactical Top-Down View</h3><p>Bearing relativ ndaj kamerës</p></div>
   <span className="subbadge">RANGE UNAVAILABLE</span></div>
  <svg role="img" aria-label="Top-down camera bearing radar" viewBox="0 0 400 245">
   {[48,92,136,172].map(radius=><polyline key={radius} points={pathFor(-90,90,radius)}
    fill="none" stroke="#31516b" strokeDasharray="3 4"/>)}
   {[-90,-60,-30,0,30,60,90].map(angle=>{
    const [x,y]=point(angle,172),[lx,ly]=point(angle,185)
    return <g key={angle}><path d={`M200 208 L${x} ${y}`} stroke="#35576f" strokeOpacity=".7"/>
     <text x={lx} y={ly} dy="-4" fill="#b2ccd9" fontSize="10" textAnchor="middle">{(angle+360)%360}°</text></g>
   })}
   {fov&&<polygon points={`200,208 ${pathFor(-fov/2,fov/2,172)}`} fill="#4fc9ed" fillOpacity=".13" stroke="#62c9e9" strokeDasharray="4 3"/>}
   {anglePoints.map(target=>{
    const [x,y]=point(target.bearing,161)
    return <g key={target.id} role="button" tabIndex={0} aria-label={target.classification+' '+target.label}
      onClick={()=>onSelect(target.id)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onSelect(target.id)}}}>
     <circle cx={x} cy={y} r={selected===target.id?7:5} fill={target.classification==='DRONE'?'#ff916b':'#62d2c5'}/>
     <text x={x} y={y+16} textAnchor="middle" fill="#d5e8f4" fontSize="9">{shortId(target.label)}</text>
    </g>
   })}
   <circle cx="200" cy="208" r="6" fill="#43baff" stroke="#e0f4ff"/><text x="200" y="231" textAnchor="middle" fill="#c3dcea" fontSize="12">Kamera</text>
  </svg>
  <p className="sg-radar-caption">{fov?'Rrezet tregojnë drejtimin. Pozicioni radial në grafik nuk është distancë.':'Fushëpamja dhe bearing janë të padisponueshme pa kalibrim të lidhur me sesionin.'}</p>
 </section>
}

export function MapObservationDetails({data,sessionId,camera,selected,live,trackHistory=[]}:Omit<Props,'onSelect'>){
 const track=data.tracks.find(t=>t.id===selected&&t.session_id===sessionId&&['DRONE','BIRD'].includes(t.classification))
 const frame=live?.state==='running'&&live.persistence_session_id===sessionId?live:null
 const liveMatch=track&&frame?.detections.find(d=>d.track_id===track.external_track_id&&d.classification===track.classification)
 const recorded=track&&[...data.detections,...trackHistory].filter(d=>d.session_id===sessionId&&d.track_id===track.id&&d.classification===track.classification)
  .sort((a,b)=>Date.parse(b.captured_at)-Date.parse(a.captured_at)||b.frame_index-a.frame_index)[0]
 // An older geographic fix cannot be presented as the location of a different live frame.
 const current=liveMatch?recorded?.frame_index===frame?.frame_index&&recorded?.captured_at===frame?.captured_at?recorded:null:recorded
 const geo=current&&data.geo.find(g=>g.session_id===sessionId&&g.detection_id===current.id&&g.classification===track?.classification&&acceptedGeo(g))
 if(!track)return <p className="sg-radar-caption">Zgjidh një objekt të gjurmuar për të parë vëzhgimin.</p>
 const confidence=liveMatch?liveMatch.confidence:recorded?.confidence
 const cells=[
  ['Class',track.classification],
  ['Confidence',confidence!=null?num(confidence*100)+'%':'Unavailable'],
  ['Track ID',track.external_track_id],
  ['Bearing relativ',liveMatch?.relative_bearing_deg!=null?num(liveMatch.relative_bearing_deg)+'°':'Unavailable'],
  ['Azimuth',liveMatch?.azimuth_deg!=null&&liveMatch.north_reference?num(liveMatch.azimuth_deg)+'° · '+liveMatch.north_reference:'Unavailable'],
  ['Elevation angle',liveMatch?.relative_elevation_deg!=null?num(liveMatch.relative_elevation_deg)+'°':'Unavailable'],
  ['Range',geo?.range_m!=null?num(geo.range_m)+' m':'Unavailable'],
  ['GPS',geo&&validCoordinate(geo.latitude,geo.longitude)?num(geo.latitude,6)+', '+num(geo.longitude,6):'Unavailable'],
  ['Measurement uncertainty',geo?num(geo.uncertainty_m)+' m · '+geo.method:'Unavailable'],
  ['Source camera',camera?.name??frame?.camera_id??'Unavailable'],
  ['Timestamp',liveMatch?frame?.captured_at??'Unavailable':recorded?.captured_at??'Unavailable'],
  ['Tracking status',liveMatch?'Confirmed on current frame':'Recorded history'],
 ]
 return <dl className="sg-map-observation">{cells.map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
}

export function TacticalPanel({data,sessionId,camera,selected,onSelect,trackHistory,live}:Props){
 const paths=cameraPixelPaths(data,sessionId,trackHistory)
 const width=Math.max(1,...paths.flatMap(path=>path.points.map(point=>point.x)))+24
 const height=Math.max(1,...paths.flatMap(path=>path.points.map(point=>point.y)))+24
 const p=useMemo(()=>tacticalPoints(data,sessionId,camera),[data,sessionId,camera]);const max=Math.max(100,...p.map(v=>v.distance+v.uncertainty_m));
 const px=(m:number)=>150+Math.max(-1,Math.min(1,m/max))*116
 const geoPaths=geographicTrackPaths(p)
 const latest=new Set(geoPaths.map(path=>path.points.at(-1)?.id))
 const labels=new Map(data.tracks.filter(track=>track.session_id===sessionId).map(track=>[track.id,track.external_track_id]))
 return <div className="tactical-view">
 <AngularRadar data={data} sessionId={sessionId} selected={selected} onSelect={onSelect} live={live}/>
 {paths.length>0&&<section className="sg-pixel-tracks">
  <h3>Trajektorja në kamerë · pikselë</h3>
  <svg role="img" aria-label="Trajektorja në kamerë në pikselë" viewBox={`0 0 ${width} ${height}`}>
   {paths.map(path=>{
    const color=path.classification==='DRONE'?'#ff916b':'#62d2c5'
    const last=path.points.at(-1)!
    return <g key={path.id} aria-label={`${path.classification} ${path.label}`}>
     {path.points.length>1&&<polyline fill="none" stroke={color} strokeWidth={path.id===selected?4:2}
      vectorEffect="non-scaling-stroke" points={path.points.map(point=>`${point.x},${point.y}`).join(' ')}/>}
     <circle cx={last.x} cy={last.y} r={Math.max(width,height)/100} fill={color}/>
    </g>
   })}
  </svg>
  <ul>{paths.map(path=><li key={path.id}><button type="button" className="sg-link" aria-pressed={selected===path.id}
   onClick={()=>onSelect(path.id)}>{path.classification} {shortId(path.label)}</button>
   <span>{path.points.length>1?`${path.points.length} kuadro`:'Në pritje të kuadrit tjetër'}</span></li>)}</ul>
 </section>}
 {p.length>0&&<><div className="panel-head compact"><div><h3>Tactical · sensor-relative</h3><p>North-up; measured position from registered geo only</p></div><span className="subbadge">{p.length} positioned</span></div>
 <div className="tactical-canvas"><svg role="img" aria-label="Sensor-relative tactical positions" viewBox="0 0 300 300" preserveAspectRatio="xMidYMid meet" className="tactical-svg">
 <defs><radialGradient id="radarFade"><stop offset="0%" stopColor="#164c66" stopOpacity="0.4"/><stop offset="100%" stopColor="#091628" stopOpacity="0"/></radialGradient></defs>
 <circle cx="150" cy="150" r="140" fill="url(#radarFade)"/>
 {[40,78,116].map(r=><circle key={r} cx="150" cy="150" r={r} stroke="#38536a" strokeOpacity=".45" fill="none" strokeDasharray="3 6"/>)}
 <path d="M150 8V292 M8 150H292" stroke="#45647b" strokeOpacity=".5" strokeDasharray="3 5"/><text x="150" y="19" fill="#819cb5" textAnchor="middle" fontSize="11">N</text><text x="285" y="145" fill="#819cb5" textAnchor="end" fontSize="11">E</text>
 {geoPaths.filter(path=>path.points.length>1).map(path=><polyline key={path.id} fill="none"
  stroke={path.points[0].classification==='DRONE'?'#ff916b':'#62d2c5'} strokeOpacity=".8"
  strokeWidth={selected===path.id?3:1.5} points={path.points.map(point=>px(point.east)+','+px(-point.north)).join(' ')}/>)}
 {p.filter(v=>latest.has(v.id)).map(v=>{const x=px(v.east),y=px(-v.north),u=Math.max(2,116*v.uncertainty_m/max);return <g key={v.id} className="tactical-target" onClick={()=>onSelect(v.track_id)} tabIndex={0} role="button" aria-label={`Select ${labels.get(v.track_id)??v.track_id}`} onKeyDown={e=>{if(e.key==='Enter')onSelect(v.track_id)}}>
 <circle cx={x} cy={y} r={Math.min(u,90)} fill={v.classification==='DRONE'?'#f97357':'#41c4b7'} fillOpacity=".09" stroke={v.classification==='DRONE'?'#f97357':'#41c4b7'} strokeOpacity=".4"/>
 <circle cx={x} cy={y} r={v.track_id===selected?7:5} fill={v.classification==='DRONE'?'#ff916b':'#62d2c5'} stroke="#081526" strokeWidth="2"/>
 <text x={x+9} y={y-7} fill="#d8e9fb" fontSize="9">{shortId(labels.get(v.track_id)??v.track_id)}</text>
 </g>})}
 <circle cx="150" cy="150" r="6" fill="#43baff" stroke="white" strokeWidth="1.5"/><path d="M150 138l-4 9h8z" fill="#43baff"/>
 </svg>
 {!p.length&&<div className="tactical-empty">No validated relative coordinates<br/><small>Camera georeference + object evidence required</small></div>}
 {p.length>0&&<span className="scale-mark">Shkalla horizontale {Math.round(max).toLocaleString()} m</span>}</div>
 <div className="geo-key"><span><i className="keydot" style={{background:'#f97357'}}/>DRONE</span><span><i className="keydot" style={{background:'#41c4b7'}}/>BIRD</span><span><i className="keydot" style={{background:'#43baff'}}/>SENSOR</span></div></>}
 </div>
}
