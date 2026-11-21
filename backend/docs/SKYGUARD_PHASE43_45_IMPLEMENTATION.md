# SkyGuard FUSION — Implementimi 43–45, versioni 0.1

**Data:** 21.11.2026. **Branch i mbrojtur:** `feature/skyguard-mvp`.

## Qëllimi dhe fushëveprimi

Kjo paketë është **shtesë** mbi SkyGuard-in ekzistues, jo zëvendësim i frontend-it, API-ve, tracker-it apo motorit të modelit. Ofron protokoll të riprodhueshëm për dataset-et dhe evidencat e Fazave 43–45. Instalimi nuk modifikon burimet ekzistuese, nuk fshin backup-e, nuk përmban videos/datasets/checkpoints dhe nuk bën `git add/commit/push`.

### Çfarë është realisht implementuar

- Inventar me SHA256 nga arkivi Halmstad, 650 video të anotuar / 650 `.mat`; 365 IR, 285 visible; 271 DRONE, 130 BIRD, 133 AIRPLANE, 116 HELICOPTER; 2 video demo jashtë GT. Ndarje train/val/test sipas **grupit të videos** (455/97/98), jo frame-level.
- Adapter MCOS (`mcos-decoder`) + OpenCV, eksport i frame-ve dhe label YOLO me 4 klasa. Rreshtat me mungesë objekti shënohen me label bosh.
- Trajnim YOLO11 me 4 klasa **si komandë reale**, jo checkpoint i fabrikuar. Trajnimi është veprim që duhet kryer në makinën me GPU.
- Klasifikim `DRONE/BIRD` me refuzim explicit `AIRPLANE/HELICOPTER/OTHER/UNKNOWN`. Default **fail closed** për legacy YOLO DRONE/BIRD me vetëm dy klasa.
- Adapter production `FourClassSkyGuardDetector` dhe wrapper `TargetOnlyTrackerProxy` që mund të përdoren në rrugën ekzistuese të shërbimit live **vetëm kur kabllimi në session manager-in ekzistues verifikohet**. Nuk aktivizohen automatikisht nga instalimi.
- Tracker referencë offline me tentative/confirmed dhe shtypje ID kur ka ambiguity. UNKNOWN nuk merr track ID.
- Për çdo frame/sesion një payload me dy projeksione: taktike dhe gjeografike, `track_id/frame_index/timestamp_ms` të përbashkëta; koordinatat WGS84 dhe distancat null pa range të jashtëm të verifikuar.
- Kalibrim threshold vetëm mbi val, me negative të grumbulluara **sipas videos** (jo korelacioni i frame-ve). Kalibrimi nuk përfshin test split.
- Benchmark YOLO real mbi test me class precision/recall/FP/FN, DRONE false-alarm në video negative, ndarje IR vs visible dhe madhësi **në piksel** (jo afër/larg në metra), latencë P50/P95/P99 dhe throughput. `yolo-map` jep mAP50/mAP50:95 nga Ultralytics.
- IDF1 për GT dhe pred IDs kur ekzistojnë; `HOTA` mbetet null/BLOCKED pa multiobject MOT GT dhe TrackEval.
- Raport pranimi `BLOCKED / REVIEW_REQUIRED / FIELD_PENDING`, pa pretendim PASS nga prania e një skedari.
- Protokoll Phase 45 `usb-field`: **vetëm me `--physical-confirm`**, lexon real USB-camera, ruan video origjinale (AVI), SHA256 dhe metrika. Nuk pretendon që ka parë dron real pa GT të pavarur.
- 25 unit tests të ekzekutuara me sukses në mjedisin e përgatitjes; kjo **nuk është** prova që suite-i i plotë i repository-t Windows është kaluar.

## Instalimi në Windows vetëm me PowerShell

1. Ruaj `SkyGuard_FUSION_43_45_Implementation.zip` në `Downloads`.
2. Në PowerShell, duke mos prekur main:

```powershell
$repo = 'C:\Users\Amigos\Projects\skyguard-fusion'
$zip = Join-Path $env:USERPROFILE 'Downloads\SkyGuard_FUSION_43_45_Implementation.zip'
Set-Location $repo
if ((git branch --show-current).Trim() -ne 'feature/skyguard-mvp') { throw 'Branch i gabuar' }
if (-not (Test-Path $zip)) { throw "Nuk gjendet: $zip" }
if (Test-Path 'backend\app\phase43_45') { throw 'Paketa ekziston; mos mbishkruaj kodin ekzistues' }
Expand-Archive -LiteralPath $zip -DestinationPath $repo
& '.\backend\.venv\Scripts\python.exe' -m pip install mcos-decoder
& '.\scripts\Run_SkyGuard_Phases43_45.ps1' -Action tests
```

Nëse PowerShell bllokon skriptet, provo komandat e CLI direkt nga `backend`; mos ndrysho globalisht politikën e sigurisë për hir të këtij instalimi. Bibliotekat `ultralytics` dhe `opencv-python` priten nga mjedisi ekzistues. Kontrollo përputhshmërinë e CUDA/PyTorch **para** instalimit/upgrade.

## Workflow sipas rendit logjik

Dataset ZIP-et duhet të jenë në `Downloads` ose përcaktoji me `-HalmstadZip` / `-DutZip` (nuk kopjohen në repo).

```powershell
# 1) Inventari real (SHA256, GT pairs, video-level split)
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action inventory

# 2) Eksport i kufizuar për kontroll kontrate; nuk mjafton për trajnim final
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action export-smoke

# 3) Eksport i videove për trajnim dhe test
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action export-full

# 4) Trajnim 4-klasësh në GPU; mund të zgjasë >1 orë
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action train -Device 0 -Epochs 35

# 5) Merr rrugën reale të best.pt nga output-i i trajnimit
$weights = 'C:\absolute\path\to\four_class\weights\best.pt'
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action calibrate -Weights $weights
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action benchmark -Weights $weights
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action yolo-map -Weights $weights -Device 0
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action acceptance
```

**Important:** `export-full` mund të prodhojë mijëra JPG dhe të kërkojë disa GB. `train` fillon realisht procesin GPU; koha e tij nuk është e garantuar. Mos e quaj fitore zyrtare një benchmark me `--max-frames` ose vetëm smoke. `train`/`calibrate`/`benchmark` nuk bëjnë commit automatik të checkpoint-it.

### Demonstrimi i Fazës 45 (vetëm në terren të kontrolluar)

```powershell
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action field-template
# Në fazën 45, me kamerë reale USB dhe dron real, me autorizimin e sigurisë:
.\scripts\Run_SkyGuard_Phases43_45.ps1 -Action usb-field -Weights $weights -CameraIndex 0 -MaxFrames 300 -PhysicalConfirm
```

Output: `raw_camera.avi`, `detections.jsonl`, `dual_scenes.jsonl`, `usb_telemetry.json`, `field_annotation_PENDING.json`. Vendos manualisht GT reale, distanca me rangefinder/RTK/tape, koordinata dhe gabim matjeje; pa to `FIELD_PENDING` është vendimi i saktë. Nuk ka funksionalitet për bllokim/jamming të dronit.

## Integrimi i kontrolluar me prodhimin ekzistues

```python
from app.phase43_45.production_adapter import FourClassSkyGuardDetector, TargetOnlyTrackerProxy
from app.detection.live_service import LiveDetectionTrackingService
from app.detection.tracker import ObjectTracker

detector = FourClassSkyGuardDetector(r'C:\absolute\four_class\weights\best.pt')
service = LiveDetectionTrackingService(detector, tracker=TargetOnlyTrackerProxy(ObjectTracker()))
```

Ky shembull verifikon **përputhshmërinë e klasave të shërbimit**, por **nuk** e lidh automatikisht `LiveCameraSession` të aplikacionit me këtë shërbim. Modifikimi i session manager-it/API-ve kërkon leximi i versionit real të kodit në Windows dhe teste regresioni; nuk duhet bërë me replace-in-blind të vargjeve. Veçanërisht dy testet historike të `ambiguous identity` duhet të zgjidhen sipas kontratës `UNKNOWN no ID`.

## Kritere që mbeten të hapura

| Kriter | Gjendja pas instalimit të paketës |
|---|---|
| Klasifikimi me 4 klasa | KOD GATI; checkpoints e trajnimi dhe GT benchmark ende PENDING |
| DRONE/BIRD/UNKNOWN në produkcion | ADAPTER GATI; lidhja me live session dhe regresionet ekzistuese PENDING |
| Multi-object ID persistence/HOTA | REFERENCË OFFLINE; GT multiobject + HOTA PENDING |
| Dy harta | KONTRATË SKENE; lidhja end-to-end me hartat ekzistuese dhe licenca e basemap PENDING |
| Range/GPS | Vetëm me matje të pavarura; përndryshe null |
| API security/RTSP SSRF, robustness | Repo ekzistues duhet të testohet me prova të pavarura; nuk pretendohet hardening i kryer |
| Live FPS | Me kamera reale dhe pajisje të dokumentuar, Faza 45 |
| Kosto dhe BOM | Duhet plotësuar me çmime burimore reale dhe hardware final |
| Dron real | FIELD_PENDING deri në provë fizike |

## Ligjshmëria, provenienca, siguria

- Halmstad ZIP: `LICENSE` CC0-1.0. README përmend përdorimin e `mcos-decoder` dhe GT MATLAB MCOS. Ky është burim i përshtatshëm për klasat negative, por nuk garanton që modeli do t'i dallojë.
- DUT ZIP: `LICENSE` Apache-2.0 për repository-n; videot GT të dataset-it janë jashtë ZIP-it. Respekto veçmas licencat/termat e shkarkimeve externe.
- Në raport ruaj SHA256 të ZIP-it, checkpoint-it dhe videove të provës. Mos publiko dataset ZIP, video personale, checkpoint apo sekrete në GitHub pa verifikuar të drejtat.
- Provat Phase45 bëhen në vend të kontrolluar dhe me autorizim për fluturim, pa rrezikuar njerëz/aeroplanë dhe pa garantuar distancat nga një kamerë monokulare.
- **Asnjë `git commit/push` nuk kryhet nga kjo paketë.**
