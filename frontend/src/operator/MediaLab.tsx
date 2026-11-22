import {useEffect,useRef,useState} from 'react'
import './media-lab.css'
import {checkedFrameCounts,ObjectCountPanel,type FrameResult} from './ObjectCountPanel'

type Point={frame:number;x_px:number;y_px:number}
type Path={track_id:string;classification:string;points:Point[]}
type Job={id:string;status:string;kind:string;frames_processed:number;session_id:string|null;error:string|null;summary?:FrameResult & {detections_per_frame_total?:Record<string,number>;conflict_review_total?:number;unique_confirmed_tracks?:Record<string,number>;mean_inference_pipeline_ms?:number|null;preview_jpeg_base64?:string|null;prediction_url?:string;model_classes?:Record<string,string>;unknown_trained?:boolean;track_paths_px?:Path[];scientific_accuracy?:string;note?:string}}
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
type LiveResult=FrameResult & {frame_index:number;session_id:string;detections:Array<{classification:string|null;track_id:string|null;confidence:number; bbox:[number,number,number,number]}>;pixel_trajectories:Array<{track_id:string;points:Point[]}>;confirmed_unique_id_count:Record<string,number>;preview_jpeg_base64?:string|null;latency_ms:number;range_m:null}
const noRange='Distanca: E PAVERIFIKUAR — GPS i kamerës nuk jep largësinë e objektit.'

class ApiError extends Error {
  readonly status:number
  constructor(status:number,message:string){
    super(message)
    this.status=status
  }
}

async function readJson(url:string,init?:RequestInit):Promise<any>{
  const r=await fetch(url,{credentials:'same-origin',cache:'no-store',...init})
  let out:any={};try{out=await r.json()}catch{ /* HTTP errors may not be JSON */ }
  if(!r.ok)throw new ApiError(r.status,typeof out?.detail==='string'?out.detail:`HTTP ${r.status}`)
  return out
}

type MediaLabProps = {
  persistenceKey?: string
  activeJobId?: string
  onJobAccepted?: (jobId?: string) => void
}

export default function MediaLab({persistenceKey = '', activeJobId, onJobAccepted}:MediaLabProps){
  const deployed=import.meta.env.VITE_SKYGUARD_DEPLOYED==='true'||import.meta.env.VITE_SKYGUARD_LOCAL_PROXY===true
  const [,setFile]=useState<File|null>(null)
  const [job,setJob]=useState<Job|null>(null)
  const [message,setMessage]=useState('')
  const [busy,setBusy]=useState(false)
  const [cameraOn,setCameraOn]=useState(false)
  const [live,setLive]=useState<LiveResult|null>(null)
  const [ipUrl,setIpUrl]=useState('')
  const [ipState,setIpState]=useState('')
  const [ipLive,setIpLive]=useState<any>(null)
  const [ipSignal,setIpSignal]=useState('')
  const ipLastFrame=useRef({index:null as number|null,receivedAt:0})
  const cameraRef=useRef<HTMLVideoElement>(null)
  const mediaStream=useRef<MediaStream|null>(null)
  const inFlight=useRef(false)
  const cameraActive=useRef(false)
  const imageCanvas=useRef<HTMLCanvasElement|null>(null)
  const jobID=job?.id??activeJobId
  const missingJobId=useRef<string|undefined>(undefined)
  useEffect(()=>{
    if(busy||!jobID||jobID===missingJobId.current||job?.status==='DONE'||job?.status==='FAILED'||job?.status==='INTERRUPTED')return
    const controller=new AbortController()
    let timer:number
    async function poll(){
      try {
        const result=await readJson('/api/media/jobs/'+jobID,{signal:controller.signal}) as Job
        if(!controller.signal.aborted)setJob(result)
      } catch(e) {
        if(!controller.signal.aborted){
          if(e instanceof ApiError&&e.status===404){
            missingJobId.current=jobID
            setJob(null)
            onJobAccepted?.()
            setMessage('The previous media job is unavailable. Choose a file to start a new analysis.')
          }else setMessage(String(e))
        }
      } finally {
        if(!controller.signal.aborted&&missingJobId.current!==jobID)timer=window.setTimeout(()=>void poll(),1200)
      }
    }
    timer=window.setTimeout(()=>void poll(),1200)
    return()=>{controller.abort();window.clearTimeout(timer)}
  },[busy,jobID,job?.status,onJobAccepted])
  useEffect(()=>()=>{
    if(cameraActive.current)void readJson('/api/media/browser/stop',{method:'POST'}).catch(()=>null)
    cameraActive.current=false
    mediaStream.current?.getTracks().forEach(t=>t.stop())
  },[])
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
    if(!ipState.startsWith('RUNNING')||(!deployed&&!persistenceKey))return
    const controller=new AbortController()
    let timer:number
    async function poll(){
      try{
        const frame=await readJson('/api/persistence/live/frame',{signal:controller.signal,headers:deployed?{}:{Authorization:'Bearer '+persistenceKey}})
        if(controller.signal.aborted)return
        const now=Date.now()
        if(frame.image_base64&&frame.frame_index!==ipLastFrame.current.index){
          ipLastFrame.current={index:frame.frame_index,receivedAt:now}
        }
        const fresh=frame.state==='running'&&frame.image_base64&&now-ipLastFrame.current.receivedAt<=10000
        setIpLive(fresh?frame:null)
        setIpSignal(fresh?'':frame.state==='running'&&now-ipLastFrame.current.receivedAt<=10000
          ?'Waiting for the first IP camera frame.'
          :'IP camera signal unavailable. Waiting for a new frame.')
      }catch(e){
        if(!controller.signal.aborted){
          setIpLive(null)
          setIpSignal('IP camera signal unavailable. '+String(e))
        }
      }finally{
        if(!controller.signal.aborted)timer=window.setTimeout(()=>void poll(),1500)
      }
    }
    void poll()
    return()=>{controller.abort();window.clearTimeout(timer)}
  },[ipState,deployed,persistenceKey])
  async function submitFile(selected:File){
    const type=selected.type.startsWith('image/')?'IMAGE':'VIDEO'
    const sizeMax=type==='IMAGE'?12*1024*1024:80*1024*1024
    if(selected.size>sizeMax){setMessage('File too large: 12MB image / 80MB video limit');return}
    setBusy(true);setMessage('Uploading to authenticated backend, actual AI required...');setJob(null)
    try{
      const out=await readJson('/api/media/upload',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Skyguard-Media-Type':type},body:selected}) as {job_id:string}
      onJobAccepted?.(out.job_id)
      const j=await readJson(`/api/media/jobs/${out.job_id}`) as Job
      setJob(j);setMessage('Backend is analyzing the uploaded file with the real checkpoint. The input and results are stored locally.')
    }catch(e){setMessage(e instanceof TypeError ? 'Backend unavailable. Start SkyGuard API on port 8000 and Vite on port 5173; check the Vite proxy and SKYGUARD_API_KEY.' : String(e))}finally{setBusy(false)}
  }
  async function startCamera(){
    try{
      setMessage('');await readJson('/api/media/browser/start',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})
      const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}},audio:false})
      mediaStream.current=stream
      if(cameraRef.current){cameraRef.current.srcObject=stream;await cameraRef.current.play()}
      cameraActive.current=true;setCameraOn(true)
    }catch(e){
      mediaStream.current?.getTracks().forEach(t=>t.stop())
      mediaStream.current=null
      if(cameraRef.current)cameraRef.current.srcObject=null
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
      if(!deployed&&!persistenceKey)throw new Error('Connect persistence in Settings before starting the IP camera')
      const headers:Record<string,string>={'Content-Type':'application/json'}
      if(!deployed)headers.Authorization='Bearer '+persistenceKey
      if(start){
        const u=new URL(ipUrl)
        if(u.protocol!=='rtsp:')throw new Error('RTSP URL required')
        await readJson('/api/persistence/live/start',{method:'POST',headers,body:JSON.stringify({source_type:'rtsp',source_location:ipUrl,fusion_enabled:false})})
        ipLastFrame.current={index:null,receivedAt:Date.now()}
        setIpSignal('Waiting for the first IP camera frame.')
        setIpState('RUNNING · server-side IP stream');setMessage('IP camera started by existing backend LiveSessionManager.')
      }else{
        await readJson('/api/persistence/live/stop',{method:'POST',headers,body:'{}'})
        setIpState('STOPPED');setIpLive(null);setIpSignal('IP camera stopped.')
      }
    }catch(e){setMessage(`IP stream: ${String(e)}`)}
  }

  const trackedLive = new Map<string, string>()
  const blockedLiveIds = new Set<string>()
  for(const detection of live?.detections??[]){
    if(!detection.track_id)continue
    if(detection.classification==='DRONE'||detection.classification==='BIRD'){
      trackedLive.set(detection.track_id,detection.classification)
    }else{
      blockedLiveIds.add(detection.track_id)
    }
  }
  const livePaths=(live?.pixel_trajectories??[]).flatMap(path=>{
    const classification=trackedLive.get(path.track_id)
    return classification&&!blockedLiveIds.has(path.track_id)?[{...path,classification}]:[]
  })
  const jobPaths=job?.summary?.track_paths_px
  const blockedJobIds=new Set(jobPaths?.filter(path=>path.classification!=='DRONE'&&path.classification!=='BIRD').map(path=>path.track_id))
  const paths=jobPaths?.filter(path=>!blockedJobIds.has(path.track_id))??livePaths
  return <section className="sg-media-lab" aria-label="AI photo video and camera analysis">
    <header><div><h2>AI Media Analysis & Live Cameras</h2><p>Real backend inference · persistence API · DRONE/BIRD tracking · UNKNOWN without IDs</p></div><span>SOFTWARE-ONLY / SCIENTIFICALLY UNCERTIFIED</span></header>
    <div className="sg-media-cols">
      <div><h3>Upload photo or video</h3><input aria-label="Choose image or video" type="file" accept="image/jpeg,image/png,video/mp4" disabled={busy} onChange={e=>{const next=e.target.files?.[0]??null;setFile(next);if(next)void submitFile(next)}}/><p>{busy?'Uploading...':'Select once to automatically analyze'}. Image: detection/classification. Video: all frames processed; track IDs are not claimed unless independently verified. Original input and annotated result are stored locally (max 80 MiB).</p></div>
      <div><h3>Laptop / USB browser camera</h3><video ref={cameraRef} muted playsInline autoPlay className="sg-camera-preview"/><button disabled={cameraOn} onClick={()=>void startCamera()}>Start browser camera</button><button disabled={!cameraOn} onClick={()=>void stopCamera()}>Stop camera</button><p>Browser captures at most 2 FPS via HTTPS or localhost; processing speed is measured, not assumed.</p></div>
      <div><h3>IP camera (RTSP, LAN)</h3><input aria-label="Private network RTSP URL" type="password" autoComplete="off" value={ipUrl} onChange={e=>setIpUrl(e.target.value)} placeholder="rtsp://192.168.x.x:554/..."/><button disabled={!ipUrl||ipState.startsWith('RUNNING')} onClick={()=>void ipAction(true)}>Start IP camera</button><button disabled={!ipState.startsWith('RUNNING')} onClick={()=>void ipAction(false)}>Stop IP camera</button><p>The backend must be able to reach the RTSP stream and have RTSP live input enabled.</p></div>
    </div>
    {message&&<p role="status" className="sg-media-message">{message}</p>}
    {job&&<div className="sg-media-result"><h3>Job: {job.status}</h3><p>Session: {job.session_id??'Media-only job'} · Frames: {job.frames_processed} · Type: {job.kind} · Error: {job.error??'none'}</p></div>}
    {live&&cameraOn&&<div className="sg-media-result"><h3>Live frame #{live.frame_index}</h3><p>Processing: {live.latency_ms} ms</p></div>}
    {ipSignal&&<p role="status" className="sg-media-message">{ipSignal}</p>}
    {ipLive&&<div className="sg-media-result"><h3>IP live: {ipLive.state}</h3><p>Frame: {ipLive.frame_index??'—'} · Observed detections: {ipLive.detections?.length??0} · Confirmed tracks: {ipLive.tracks?.length??0}</p></div>}
    {job?.status==='DONE'&&job.kind==='VIDEO'&&job.summary?.prediction_url&&<div className="sg-media-result"><h3>Annotated analysis video</h3><video className="sg-media-output" controls preload="metadata" src={job.summary.prediction_url}/><a href={job.summary.prediction_url} download>Open annotated video</a></div>}
    {job?.summary?.preview_jpeg_base64&&<img className="sg-media-output" alt="Annotated upload" src={'data:image/jpeg;base64,'+job.summary.preview_jpeg_base64}/>}
    {cameraOn&&live?.preview_jpeg_base64&&<img className="sg-media-output" alt="Annotated browser camera frame" src={'data:image/jpeg;base64,'+live.preview_jpeg_base64}/>}
    {ipLive?.image_base64&&<img className="sg-media-output" alt="Annotated IP camera frame" src={'data:image/jpeg;base64,'+ipLive.image_base64}/>}
    {cameraOn&&live&&<ObjectCountPanel counts={checkedFrameCounts(live)} heading="Browser camera frame"/>}
    {typeof job?.summary?.conflict_review_total==="number"&&job.summary.conflict_review_total>0&&<p>Conflicting proposals for review: {job.summary.conflict_review_total}</p>}
    {job?.kind==="IMAGE"&&job.status==="DONE"&&<ObjectCountPanel counts={checkedFrameCounts(job.summary??{})} heading={checkedFrameCounts(job.summary??{})?"ANALYSIS COMPLETE":"Image analysis"}/>}
    {paths.length>0&&<details><summary>Recorded pixel trajectories ({paths.length})</summary><TrajectoryPlot paths={paths}/><div className="sg-media-tracks">{paths.slice(0,30).map(p=><div key={p.track_id}><strong>{p.track_id}</strong> · {('classification' in p ? p.classification : 'DRONE/BIRD')} · {p.points.length} measured frame positions · Last: {JSON.stringify(p.points[p.points.length-1])}</div>)}</div></details>}
    <p className="sg-media-caveat">{noRange} UNKNOWN never gets track ID or trajectory. A photo has no motion history. RGB classification and model checkpoints still require source-disjoint ground-truth validation.</p>
  </section>
}
