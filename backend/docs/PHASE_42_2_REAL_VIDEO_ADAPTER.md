# SkyGuard FUSION — Phase 42.2: real-video YOLO baseline → offline evaluator

## Qëllimi dhe çfarë nuk provon

Adapteri **lexon video lokale**, ekzekuton modelin lokal `YOLO.track` frame-pas-frame, ruan rezultate me `track_id` të tracker-it kur ekzistojnë, i bashkon vetëm me **ground truth të krijuar në mënyrë të pavarur**, dhe i kalon në algoritmin e Fazës 42.1. **Nuk** përdor kamerë live, nuk shkarkon dataset-e/model dhe nuk e ndryshon backend-in e prodhimit.

**Dallim kritik:** Kjo është matje e *Ultralytics YOLO + ByteTrack/BoT-SORT baseline*. Nuk është ende test end-to-end i tracker-it `SkyGuard LiveSessionManager`; ajo krahasim është objektiv i Fazës 42.3. FPS i manifestit mat thirrjet e inferencës (dhe një tregues me decoding), **jo** FPS-in e aplikacionit live. Asnjë rezultat nuk është GPS, distancë apo identifikim fizik i konfirmuar i dronit.

## Parakushtet

- `backend/.venv/Scripts/python.exe` me `ultralytics` dhe `cv2` të instaluara **më parë**. Asgjë nuk instalohet automatikisht.
- Video lokale e ligjshme (`mp4/mov/avi/mkv/m4v`), pa URL RTSP ose cloud.
- Checkpoint lokal `.pt` nga burim i besuar: **mos hapni checkpoint-e të panjohura**, sepse ngarkimi PyTorch mund të ketë rreziqe ekzekutimi kodi.
- JSONL ground truth nga anotues të pavarur, pa predictions nga modeli. Duhet të verifikohen licencat dhe termat e çdo dataseti.
- JSON mapim i emrave realë `model.names` në klasa të unifikuara; shih `backend/examples/phase42_2_class_map_EXAMPLE.json`. Mapimi duhet të përfshijë *të gjitha* klasat reale të checkpoint-it, pa emra të shtuar.

## Formati ground truth (vetëm anotime, zero-based)

```json
{"sequence_id":"clip_001","frame_index":0,"width":640,"height":480,"ground_truth":[{"id":"drone_real_001","class":"drone","bbox_xyxy":[15,25,55,65]}]}
{"sequence_id":"clip_001","frame_index":1,"width":640,"height":480,"ground_truth":[]}
```

- `id` e ground truth është identitet i anotuar në video, jo track ID nga AI, GPS apo numër serie droni.
- Shtoni edhe frame-et pa objekte: `ground_truth: []`, që të maten false positives.
- `frame_index` nis me `0`; gjerësia/lartësia duhet të përputhen me videon origjinale **pa ndryshim shkalle**.
- Pa `--allow-partial`, duhet të anotohen të gjitha frame-et e videos. Me `--allow-partial`, vetëm frame-et e anotuar vlerësohen, ndërsa YOLO *prapëseprapë* përpunon çdo frame për të ruajtur vazhdimësinë e track-ut. Në raport shënohet përqindja e mbulimit. Mos raportoni precision/recall të pjesshëm si për gjithë videon.
- Nuk pranohet fusha `predictions` në anotime: ajo prodhohet ekskluzivisht nga inferenca lokale.

## Përdorimi në PowerShell

Kjo është vetëm **mostër komandash**: vendosni video dhe anotime reale, pas kontrollit të licencës; mos përdorni fixture-in sintetik 42.1 si ground truth për rezultate të jashtme.

```powershell
cd C:\Users\Amigos\Projects\skyguard-fusion
$env:PYTHONPATH = (Resolve-Path 'backend').Path
$weights = 'backend\runs\skyguard\drone_bird_smoke\weights\best.pt'
$video = 'C:\PATH\TO\licensed_labeled_video.mp4'
$gt = 'C:\PATH\TO\independent_ground_truth.jsonl'
$map = 'C:\PATH\TO\class_map.json'
$out = Join-Path $env:USERPROFILE ('Downloads\SkyGuard_42_2_Real_' + (Get-Date -Format yyyyMMdd_HHmmss))

& 'backend\.venv\Scripts\python.exe' -m app.evaluation.real_video_adapter `
  --video $video --weights $weights --ground-truth $gt `
  --class-map $map --sequence-id clip_001 --output-dir $out `
  --max-frames 3000 --confidence 0.25 --iou-threshold 0.5 `
  --imgsz 640 --device cpu --tracker bytetrack.yaml `
  --confirm-license --confirm-trusted-checkpoint

if ($LASTEXITCODE -ne 0) { throw 'Real-video benchmark did not complete' }
```

Për të testuar edhe frame-e të paanotuara gjatë inferencës dhe të vlerësoni vetëm të anotuarat, shtoni `--allow-partial` me vetëdije. `--device 0` kërkon GPU/torch CUDA të disponueshëm.

## Skedarët e raportit

- `annotated_predictions.jsonl`: input-i i rregullt për evaluator-in 42.1, me parashikimet **e modelit**, jo parashikime të sajuara.
- `metrics.json`: precision/recall/F1, confusion, ID switches, maksimumi i vërejtur i dronëve në fragmentin e matur. **Maksimumi i vërejtur ≠ kapaciteti maksimal i garantuar**.
- `manifest.json`: SHA-256 i videos, checkpoint-it, anotimeve, class map; parametra inferencë, mbulim anotimesh, kohë inferencash dhe metadata/versione.

Rezultatet nuk mbishkruhen: `--output-dir` duhet të jetë rrugë që nuk ekziston. Nëse procedura dështon gjatë krijimit të raportit, pastron vetëm dosjen që sapo ka krijuar.

## Kriteret që mbeten të hapura

1. Testet 42.2 me video reale të licencuara dhe GT të verifikuar nuk kryhen automatikisht nga skripti i instalimit.
2. Ndërrimet e ID-së së tracker-it të *prodhimit SkyGuard*, skenarët kompleksë të kryqëzimit, durueshmëria dhe kapaciteti me shumë dronë kërkojnë 42.3.
3. Performanca e kamerës live, mjedisi i ndriçimit, konfigurimi CUDA, dhe matjet end-to-end kërkojnë 42.4.
4. Një kamerë monokulare pa metodë të validuar për thellësinë nuk mat largësinë/GPS e dronit. Raportet 42.2 nuk duhet të përdoren për pretendime gjeografike.
