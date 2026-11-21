import {useEffect,useRef,useState} from 'react'
import './media-lab.css'

type Point={frame:number;x_px:number;y_px:number}
type Path={track_id:string;classification:string;points:Point[]}
type Job={id:string;status:string;kind:string;frames_processed:number;session_id:string|null;error:string|null;summary?:{detections_per_frame_total?:Record<string,number>;unique_confirmed_tracks?:Record<string,number>;mean_inference_pipeline_ms?:number|null;preview_jpeg_base64?:string|null;track_paths_px?:Path[];scientific_accuracy?:string;note?:string}}
function TrajectoryPlot({paths}:{paths:Path[]}){
  const points=paths.flatMap(p=>p.points)
  if(!points.length)return null
  const xmin=Math.min(...points.map(p=>p.x_px)),xmax=Math.max(...points.map(p=>p.x_px))
  const ymin=Math.min(...points.map(p=>p.y_px)),ymax=Math.max(...points.map(p=>p.y_px))
  const dx=Math.max(1,xmax-xmin),dy=Math.max(1,ymax-ymin)
  const x=(n:number)=>8+(n-xmin)/dx*84
  const y=(n:number)=>8+(n-ymin)/dy*84
  return <div><p>Track motion history: relative image-plane coordinates (pixels), NOT geographical GPS.</p>
  <svg className="sg-media-trajectory" viewBox="0 0 100 100" role="img" aria-label="Tracked objects' measured image-plane pixel movement trajectories">
  <rect x="0" y="0" width="100" height="100" fill="#061421"/>
  {paths.slice(0,20).map((p,i)=><g key={p.track_id}>
    <polyline points={p.points.map(t=>`${x(t.x_px)},${y(t.y_px)}`).join(' ')} fill="none" stroke={i%2?'#e9a65a':'#4ee7d4'} strokeWidth=".9" vectorEffect="non-scaling-stroke"/>
    {p.points.length>0&&<circle cx={x(p.points[p.points.length-1].x_px)} cy={y(p.points[p.points.length-1].y_px)} r="1.3" fill={i%2?'#e9a65a':'#4ee7d4'}/>}
  </g>)}
  </svg></div>
}
type LiveResult={frame_index:number;session_id:string;detections:Array<{classification:string;track_id:string|null;confidence:number; bbox:[number,number,number,number]}>;pixel_trajectories:Array<{track_id:string;points:Point[]}>;confirmed_unique_id_count:Record<string,number>;preview_jpeg_base64?:string|null;latency_ms:number;range_m:null}
const noRange='Distanca: E PAVERIFIKUAR — GPS i kamerës nuk jep largësinë e objektit.'

async function readJson(url:string,init?:RequestInit):Promise<any>{
  const r=await fetch(url,{credentials:'same-origin',cache:'no-store',...init})
  let out:any={};try{out=await r.json()}catch{ /* HTTP errors may not be JSON */ }
  if(!r.ok)throw new Error(typeof out?.detail==='string'?out.detail:`HTTP ${r.status}`)
  return out
}

export default function MediaLab(){
  const [file,setFile]=useState<File|null>(null)
  const [job,setJob]=useState<Job|null>(null)
  const [message,setMessage]=useState('')
  const [busy,setBusy]=useState(false)
  const [cameraOn,setCameraOn]=useState(false)
  const [live,setLive]=useState<LiveResult|null>(null)
  const [ipUrl,setIpUrl]=useState('')
  const [ipState,setIpState]=useState('')
  const [ipLive,setIpLive]=useState<any>(null)
  const cameraRef=useRef<HTMLVideoElement>(null)
  const mediaStream=useRef<MediaStream|null>(null)
  const inFlight=useRef(false)
  const cameraActive=useRef(false)
  const imageCanvas=useRef<HTMLCanvasElement|null>(null)
  const jobID=job?.id
  useEffect(()=>{
    if(!jobID||job?.status==='DONE'||job?.status==='FAILED'||job?.status==='INTERRUPTED')return
    const timer=window.setInterval(()=>{void readJson(`/api/media/jobs/${jobID}`).then(setJob).catch(e=>setMessage(String(e)))},1200)
    return()=>window.clearInterval(timer)
  },[jobID,job?.status])
  useEffect(()=>()=>{cameraActive.current=false;mediaStream.current?.getTracks().forEach(t=>t.stop())},[])
  useEffect(()=>{
    if(!cameraOn)return
    let stopped=false
    async function tick(){
      if(stopped||!cameraActive.current||inFlight.current)return
      const v=cameraRef.current
      if(!v||v.videoWidth<16||v.videoHeight<16)return
      inFlight.current=true
      try{
        const canvas=imageCanvas.current??document.createElement('canvas');imageCanvas.current=canvas
        const scale=Math.min(1,640/Math.max(v.videoWidth,v.videoHeight))
        canvas.width=Math.round(v.videoWidth*scale);canvas.height=Math.round(v.videoHeight*scale)
        const context=canvas.getContext('2d');if(!context)throw new Error('Canvas unavailable')
        context.drawImage(v,0,0,canvas.width,canvas.height)
        const blob=await new Promise<Blob|null>(resolve=>canvas.toBlob(resolve,'image/jpeg',0.68))
        if(!blob)throw new Error('Unable to encode camera frame')
        const data=await readJson('/api/media/browser/frame',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob}) as LiveResult
        if(!stopped)setLive(data)
      }catch(e){if(!stopped)setMessage(`Camera: ${String(e)}`)}
      finally{inFlight.current=false}
    }
    const timer=window.setInterval(()=>{void tick()},500) // at most 2 FPS, not a measured capability claim
    return()=>{stopped=true;window.clearInterval(timer)}
  },[cameraOn])
  useEffect(()=>{
    if(!ipState.startsWith('RUNNING'))return
    const timer=window.setInterval(()=>{void readJson('/api/persistence/live/frame').then(setIpLive).catch(e=>setMessage(String(e)))},1500)
    return()=>window.clearInterval(timer)
  },[ipState])
  async function submitFile(){
    if(!file)return
    const type=file.type.startsWith('image/')?'IMAGE':'VIDEO'
    const sizeMax=type==='IMAGE'?12*1024*1024:40*1024*1024
    if(file.size>sizeMax){setMessage('File too large: 12MB image / 40MB video limit');return}
    setBusy(true);setMessage('Uploading to authenticated backend, actual AI required...');setJob(null)
    try{
      const out=await readJson('/api/media/upload',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Skyguard-Media-Type':type},body:file}) as {job_id:string}
      const j=await readJson(`/api/media/jobs/${out.job_id}`) as Job
      setJob(j);setMessage('Backend job created. Results are stored in PostgreSQL.')
    }catch(e){setMessage(String(e))}finally{setBusy(false)}
  }
  async function startCamera(){
    try{
      setMessage('');await readJson('/api/media/browser/start',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
      const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}},audio:false})
      mediaStream.current=stream
      if(cameraRef.current){cameraRef.current.srcObject=stream;await cameraRef.current.play()}
      cameraActive.current=true;setCameraOn(true)
    }catch(e){
      void readJson('/api/media/browser/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).catch(()=>null)
      setMessage(`Camera unavailable: ${String(e)}`)
    }
  }
  async function stopCamera(){
    cameraActive.current=false;setCameraOn(false)
    mediaStream.current?.getTracks().forEach(t=>t.stop());mediaStream.current=null
    if(cameraRef.current)cameraRef.current.srcObject=null
    try{await readJson('/api/media/browser/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})}catch(e){setMessage(String(e))}
  }
  async function ipAction(start:boolean){
    try{
      if(start){
        const u=new URL(ipUrl)
        if(u.protocol!=='rtsp:')throw new Error('RTSP URL required')
        await readJson('/api/persistence/live/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_type:'rtsp',source_location:ipUrl,fusion_enabled:false})})
        setIpState('RUNNING · server-side IP stream');setMessage('IP camera started by existing backend LiveSessionManager.')
      }else{
        await readJson('/api/persistence/live/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
        setIpState('STOPPED');setIpLive(null)
      }
    }catch(e){setMessage(`IP stream: ${String(e)}`)}
  }
  const image=job?.summary?.preview_jpeg_base64??live?.preview_jpeg_base64??ipLive?.image_base64
  const paths=job?.summary?.track_paths_px??live?.pixel_trajectories?.map(p=>({...p,classification:'DRONE/BIRD'}))??[]
  return <section className="sg-media-lab" aria-label="AI photo video and camera analysis">
    <header><div><h2>AI Media Analysis & Live Cameras</h2><p>Real backend inference · PostgreSQL evidence · DRONE/BIRD tracking · UNKNOWN without IDs</p></div><span>SOFTWARE-ONLY / SCIENTIFICALLY UNCERTIFIED</span></header>
    <div className="sg-media-cols">
      <div><h3>Upload photo or video</h3><input aria-label="Choose image or video" type="file" accept="image/jpeg,image/png,video/mp4,video/x-msvideo,video/webm" onChange={e=>setFile(e.target.files?.[0]??null)}/><button disabled={!file||busy} onClick={()=>void submitFile()}>{busy?'Uploading...':'Analyze with SkyGuard AI'}</button><p>Photo: detection/classification only. Video: per-frame detections, persistent track IDs and pixel trajectories. Max 120 frames; original file is stored server-side.</p></div>
      <div><h3>Laptop / USB browser camera</h3><video ref={cameraRef} muted playsInline autoPlay className="sg-camera-preview"/><button disabled={cameraOn} onClick={()=>void startCamera()}>Start browser camera</button><button disabled={!cameraOn} onClick={()=>void stopCamera()}>Stop camera</button><p>Browser captures at most 2 FPS via HTTPS or localhost; processing speed is measured, not assumed.</p></div>
      <div><h3>IP camera (RTSP, LAN)</h3><input aria-label="Private network RTSP URL" type="password" autoComplete="off" value={ipUrl} onChange={e=>setIpUrl(e.target.value)} placeholder="rtsp://192.168.x.x:554/..."/><button disabled={!ipUrl||ipState.startsWith('RUNNING')} onClick={()=>void ipAction(true)}>Start IP camera</button><button disabled={!ipState.startsWith('RUNNING')} onClick={()=>void ipAction(false)}>Stop IP camera</button><p>RTSP reachable from backend host/container required; only private LAN IPv4 addresses accepted.</p></div>
    </div>
    {message&&<p role="status" className="sg-media-message">{message}</p>}
    {job&&<div className="sg-media-result"><h3>Job: {job.status}</h3><p>Session: {job.session_id??'pending'} · Frames: {job.frames_processed} · Type: {job.kind} · Error: {job.error??'none'}</p></div>}
    {live&&cameraOn&&<div className="sg-media-result"><h3>Live frame #{live.frame_index}</h3><p>Detections this frame: {live.detections.length} · Confirmed identities: {JSON.stringify(live.confirmed_unique_id_count)} · Processing: {live.latency_ms} ms</p></div>}
    {ipLive&&<div className="sg-media-result"><h3>IP live: {ipLive.state}</h3><p>Frame: {ipLive.frame_index??'—'} · Observed detections: {ipLive.detections?.length??0} · Confirmed tracks: {ipLive.tracks?.length??0}</p></div>}
    {image&&<img className="sg-media-output" alt="AI annotated image from the authenticated backend, not a simulated overlay" src={`data:image/jpeg;base64,${image}`}/>}
    {job?.summary&&<div className="sg-media-result"><p>Detection events: {JSON.stringify(job.summary.detections_per_frame_total??{})}</p><p>Distinct confirmed track IDs: {JSON.stringify(job.summary.unique_confirmed_tracks??{})}</p><p>Measured mean pipeline latency: {job.summary.mean_inference_pipeline_ms??'—'} ms</p><p>Ground-truth accuracy: {job.summary.scientific_accuracy??'NOT MEASURED'}</p></div>}
    {paths.length>0&&<details><summary>Recorded pixel trajectories ({paths.length})</summary><TrajectoryPlot paths={paths}/><div className="sg-media-tracks">{paths.slice(0,30).map(p=><div key={p.track_id}><strong>{p.track_id}</strong> · {('classification' in p ? p.classification : 'DRONE/BIRD')} · {p.points.length} measured frame positions · Last: {JSON.stringify(p.points[p.points.length-1])}</div>)}</div></details>}
    <p className="sg-media-caveat">{noRange} UNKNOWN never gets track ID or trajectory. A photo has no motion history. RGB classification and model checkpoints still require source-disjoint ground-truth validation.</p>
  </section>
}
