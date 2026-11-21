# SkyGuard FUSION — Technical State, Scientific Acceptance & Remaining Roadmap

**Data e referencës:** 21 nëntor 2026 · **Branch:** `feature/skyguard-mvp` · **Commit i fundit i verifikuar nga logu i përdoruesit:** `446a7e519e37f02553abe826871051e9489316ad` (para paketës V6).

> **Status epistemik:** Ky raport dallon (A) kod të pranishëm/ të publikuar, (B) teste dhe rezultate të konfirmuara nga logjet e përdoruesit, dhe (C) elemente ende pa dëshmi. Është raport inxhinierik i progresit, **jo certifikatë se sfida është përmbushur 100%**. Pragjet përfundimtare numerike duhet të merren nga dokumenti zyrtar i sfidës; në mungesë të tij këtu nuk shpikim FPS, metra, saktësi, buxhete ose pikë jurie.

## 1. Objektivi i sistemit

SkyGuard FUSION synon të monitorojë një burim vizual, të dallojë `DRONE`, `BIRD` dhe `UNKNOWN`, të ruajë identitetin dhe trajektoret **vetëm** për `DRONE/BIRD`, të dokumentojë rezultatet dhe burimet e provave në PostgreSQL, të paraqesë informacionin në Operator Console dhe në dy harta të sinkronizuara (gjeografike dhe taktike), dhe të raportojë performancën/paqartësinë pa metrika të shpikura.

**Rregulla të panegociueshme:** `UNKNOWN` nuk merr track ID, trajektore apo gjeolokacion. Një kamerë monokulare pa kalibrim dhe referencë nuk jep GPS/distance të saktë. DEMO dhe SIMULATION nuk kanë peshë si evidencë e testeve reale. Provimet fizike me USB-kamerë dhe dron real bëhen **në Fazën 45**, në mjedis të kontrolluar dhe në përputhje me rregullat e sigurisë.

## 2. Çfarë është realizuar dhe çfarë është vërtetuar

| Fusha | Implementimi i pranishëm në source | Evidenca e konfirmuar | Kufizimi real |
|---|---|---|---|
| Docker/BuildKit | Backend, migration, gateway, Compose | Image build CPU-only, `torch 2.6.0+cpu` me CUDA False sipas logut | Nuk mat FPS për skena reale; GPU nuk është aktiv |
| PostgreSQL | ORM SQLAlchemy, migrime Alembic `0001`, `0002` | `postgres:17-bookworm` healthy; schema `0002` verifikuar | Përmirësim backup/restore dhe teste DB nën ngarkesë ende duhen |
| Backend | FastAPI, API persistence, live integration, media endpoints | Uvicorn në `0.0.0.0:8000` i nisur | Efektiviteti i modelit dhe integrimi plotësisht live duhet matur |
| Gateway | Auth operator me hash PBKDF2, cookie HttpOnly, proxy API | `/healthz`, faqja login dhe hyrja e re verifikuar; porta loopback `8088` | HTTP lokal, pa TLS publik, single-operator jo RBAC ndërmarrjeje |
| Frontend | React/TypeScript Operator Console; faqe monitoring, tracking, alerts, maps, evidence, performance | Build dhe login të funksionalizuara në deployment; përdoruesi konfirmoi hyrjen | Renderimi i të dhënave reale pas lidhjes me DB kërkon provë E2E dhe kontroll vizual |
| Persistenca | Kamera, sesione, frames, detections, tracks, alerts, audit, benchmark, evidence, performance | Migrimet funksionuan | Duhet verifikuar API GET dhe UI me raste komplekse, jo vetëm schema |
| Klasifikimi | Pipeline DRONE/BIRD/UNKNOWN dhe kode për stërvitje/adapter | Kod ekziston në source të commit-it `446a7e5` | **S'ka matje të provuara këtu** të precision/recall në holdout; prioritet urgjent |
| Tracking | ID për DRONE/BIRD, histori vëzhgimesh dhe kufizime DB për UNKNOWN | Kontrata dhe skema të pranishme | IDF1/HOTA, ID switches, crossing/occlusion ende pa protokoll të plotë raportimi |
| Harta të dyfishta | MapLibre / geospatial & top-down në frontend/kit | Kod i pranishëm | Sinkronizimi në burim real, kalibrim, saktësia dhe uncertainty ende pa test me GT |
| Video dhe dataset-e | Media upload/processing, video research tooling | Module/skripte të pranishme | Procesimi demonstrues nuk është provë e performancës së detektorit |
| GitHub | Source + konfigurim lokal pa sekrete | Commit `446a7e5` i push-uar në `feature/skyguard-mvp` | Katër file të pakomituara në logun para V6; V6 duhet ekzekutuar lokalisht |
| Kamera USB/dron | Integrimi dhe kontratat në kod | **Jo të validuara fizikisht** | Faza 45 e detyrueshme |

**Kujdes:** Statusi `healthy` i gateway nuk vërteton vetë PostgreSQL: endpoint-i `/healthz` deklaron `database: not_checked`. Duhet `/api/persistence/health` dhe GET me sesion operatori, sikurse në runner-in E2E të kësaj pakete.

### Pikat e rregulluara teknikisht më 21 nëntor

1. BuildKit kompatibël me `linux/amd64` për të evituar `exec format error`.
2. Backend/migration me PyTorch CPU-only që shmang paketat e mëdha CUDA në këtë kompjuter.
3. `postgres:17-bookworm` në vend të variantit Alpine që nuk ekzekutohej në Docker-in lokal.
4. Migrimet `0001` dhe `0002` të zbatuara në volume persistente, pa `down -v` apo `prune`.
5. Gateway i publikuar **vetëm** në `127.0.0.1:8088` me konfigurim kompatibiliteti.
6. Reset i sigurt i `operator`, me hash dhe çelës sesioni të rinovuar; testimi i hyrjes kaloi.
7. Source dhe konfigurimi pa sekrete u publikuan në branch-in e veçantë përmes GitHub. `main` mbeti i paprekur.

**Infrastrukturë ≠ sukses shkencor.** Asnjë prej këtyre pikave nuk nënkupton automatikisht klasifikim të saktë, distanca të kalibruara apo performancë të garantuar.

## 3. Arkitektura e zgjedhur

```text
Browser (operator login + console)
  └─ HTTP 127.0.0.1:8088, cookie HttpOnly, single operator
      └─ Gateway FastAPI (origjina / auth / lejimet e API-së)
          └─ Backend FastAPI (:8000, vetëm rrjet Docker)
              ├─ Perception: detector + classifier → DRONE / BIRD / UNKNOWN
              ├─ Tracking: ID persistence vetëm DRONE/BIRD
              ├─ Media: video/image/browser input
              ├─ Geometry: vetëm prova të kalibruara + uncertainty
              └─ Persistence API / Store (SQLAlchemy)
                  └─ PostgreSQL 17 → Alembic 0002
```

Mjedisi i Docker-it lokal ka `backend`, `gateway`, `migration`, `postgres`, dhe `caddy` që nuk është aktiv në variantin aktual. Dy konfigurimet kompatibile (Postgres Debian dhe gateway loopback) ekzistojnë jashtë Git; repository përmban udhëzime dhe `deploy/compose.local-compat.yaml` për riprodhim. Kujdes me nisjen vetëm të `deploy/compose.yaml`: mund të kthejë Postgres te Alpine dhe të prishë hapjen e shërbimit. Konfirmo `docker compose config` përpara çdo deploy-i tjetër.

## 4. DEMO data dhe prova PostgreSQL → Backend → Gateway → Frontend

### Çfarë ofron kjo paketë

- `backend/scripts/seed_skyguard_demo.py`: ruan **vetëm një dataset të etiketuar** `SKYGUARD_DEMO_FIXTURE_V1`, kamera `[DEMO] Simulated observation camera` (`SIMULATED`), sesion `SIMULATION/SOFTWARE_ONLY`, **5 frames**, **15 detections** (5 DRONE, 5 BIRD, 5 UNKNOWN), **2 tracks** (DRONE, BIRD) dhe **2 alerts** `[DEMO ONLY]`.
- Idempotent: ekzekutimi i dytë nuk shumëfishon vëzhgimet.
- **Nuk krijon** `GeoEvidence`, performancë FPS/latency, benchmark, media apo GT të rreme. UNKNOWN gjithmonë `track_id=null`.
- `scripts/Run_SkyGuard_Demo_Seed.ps1`: kontrollon backend/gateway, verifikon Alembic, bën seed me `-Seed`, dhe opsionalisht provon HTTP logimin dhe GET-et me `-VerifyGateway` duke kërkuar fjalëkalimin në mënyrë të fshehtë.

### Komandat (pas instalimit të kit-it në repo)

```powershell
cd C:\Users\nistr\skyguard-fusion
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Run_SkyGuard_Demo_Seed.ps1 -Repository 'C:\Users\nistr\skyguard-fusion'
# Kalon vetëm preflight, pa ndryshuar databazën.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Run_SkyGuard_Demo_Seed.ps1 -Repository 'C:\Users\nistr\skyguard-fusion' -Seed -VerifyGateway
# Kërkon password operator lokal për testin real HTTP. Mos e publikoni.
```

Pas `SKYGUARD_AUTHENTICATED_GATEWAY_BACKEND_POSTGRES_E2E_VERIFIED`, hap `http://127.0.0.1:8088/operator.html`, bëj refresh dhe zgjidh sesionin `SIMULATION`. Kontrollo **Command Center, Tracking, Alerts, Cameras, Evidence, Maps**. UI duhet të tregojë numrat e DEMO-s, por me etiketa të dukshme që nuk janë prova reale. Nëse faqja mbetet `DISCONNECTED`, është problem i integrimit të frontend-it me cookie gateway, i cili duhet rregulluar dhe **gateway duhet ribërë** për React build-in e ri. Mos fut `SKYGUARD_DB_API_KEY` në browser; shërbimi është në gateway.

#### Tabela e provave E2E që duhen regjistruar

| Kontroll | Rezultat i pritshëm | Çfarë provon |
|---|---|---|
| `GET /auth/session` pas login | `authenticated: true` | Siguri sesioni |
| `GET /api/persistence/cameras` | kamera `[DEMO]` | Gateway → backend → SQL |
| `GET /api/persistence/sessions` | 1 sesion DEMO | Persistencë dhe listim |
| `GET /api/persistence/detections?session_id=...` | 15 | Lidhje frame/detection |
| `GET /api/persistence/tracks?session_id=...` | 2 | DRONE/BIRD ID vetëm |
| UNKNOWN detections | 5 me `track_id = null` | Rregull i detyrueshëm |
| `GET /api/persistence/alerts?...` | 2 të etiketuara DEMO | Alert management |
| `GET /api/persistence/sessions/.../summary` | 5 frames | Raportim i saktë |
| Geospatial/performance | 0 të dhëna | Pa GPS/FPS të shpikura |
| UI vizual | tabelat shfaqin të njëjtat numra | Integrim i vërtetë React ↔ API |

**Ndarja e provave:** Seed-i provon zinxhirin e databazës dhe API-së, **jo** cilësinë e modelit të AI. Me 5 frames sintetikë nuk raportohen precision/recall, FPS real, range, apo matje në terren.

## 5. Roadmap i mbetur, sipas varësive teknike

### Faza 42 — Klasifikimi funksional dhe lidhja me dataset-et (prioritet P0)

**Punët:** audit i checkpoint-it `best.pt` (path/sha/model classes), verifikim i mapping DRONE/BIRD/UNKNOWN, input preprocessing, confidence calibration, false-positive gating, trajtim explicit i airplane/helicopter si UNKNOWN, integrim me video/image endpoint. Kontroll i njësive, dtype, RGB/BGR, përmasave dhe versioneve Ultralytics/PyTorch. Datasets të ndahen sipas sekuencës/kamerës/video origjinës për të evituar leakage.

**Dalja e matshme:** skript unik që jep dataset hash, checkpoint hash, per-class confusion matrix, precision/recall/F1, false-positive për kategori distractor, shembuj failure dhe urdhra reproducible. Një classifier që jep gjithmonë një klasë pa kalibrim të uncertainty nuk e kalon kriterin.

### Faza 43 — Video multi-object tracking, identiteti dhe frontend/database E2E

**Punët:** test me multi-object scenes, occlusion, crossing, target re-entry; IDF1/HOTA, switches, fragmentation; event contracts JSON; validim të transaksioneve frames/detections/alerts, query paging; UI demo fixture dhe më pas dataset real video. Integrimi cookie gateway dhe React auto-connect duhet të jetë real dhe i testuar. Alert acknowledgement dhe audit trail të provohen përmes API-së.

**Dalja e matshme:** 1 scenario file me referenca dhe hash + API replay + raporte metrikash + browser E2E; `UNKNOWN` i refuzuar për tracking dhe trajektore në runtime, database dhe UI. Testet e regresionit duhet të kapin shkeljen.

### Faza 44 — Harta, range me uncertainty, performanca dhe prova të pavarura

**Punët:** harta WGS84 nga burim i licencuar, harta top-down sinkron me të njëjtat track IDs, timestamp, trajectory; regjistrim i calibration, extrinsics/intrinsics, metoda triangulimi ose reference measurement; error bars, confidence intervals, no-location fallback; FPS dhe latency p50/p95/p99 me kohë monotone dhe hardware report; disk/network budget; kontroll i sigurisë, auth, API schema, path traversal dhe dokumentim për gjykim black-box.

**Dalja e matshme:** video me ground truth të koordinatave/distance, gabimi MAE/RMSE dhe coverage të intervaleve, test sinkronizimi i hartave, raporte performance CPU (ose GPU nëse autorizohet), teste negative e adversarial. Pa GT, paraqit `N/A`, jo numra të shpikur.

### Faza 45 — Validimi fizik USB + dron real, finale për jurinë

**Punët:** përdor kamerë USB reale dhe dron real me autorizim dhe masat e sigurisë; mat ground truth me metodë të dokumentuar; skena me zogj/distractors kur lejohet, distanca/ndriçim/moti të kontrolluara; verifiko latency, FPS, detection, tracking ID, llogarit gabimin spatial dhe uncertainty; siguro raportim të gabimeve dhe dështimeve. Bëj setup nga repository i pastër me komandat e dokumentuara, pa lokal secret në git. Video evidence i paedituar + checksums + raporte testesh.

**Dalja e matshme:** raport final signed-by-team me hardware, software version, data, methods, të gjitha limitet, video evidence, confusion matrix, ID metrics, error bars, FPS, latency, acceptance table, reproducibility dhe release tag. `PHYSICAL_VERIFIED` vetëm pas këtij testi, jo nga dataset video.

### Gate përfundimtar objektiv

Për secilin kriter në dokumentin zyrtar të sfidës kërko: (1) ID/kërkesë, (2) implementim konkret, (3) rast testimi, (4) evidencë bruto dhe hash, (5) rezultat numerik dhe kufiri i lejuar, (6) `PASS/FAIL/NOT_TESTED`, (7) pronari i testit dhe mjedisi. Kriteri vlerësohet `PASS` vetëm kur të gjitha provat ekzistojnë dhe kalojnë **pragjet reale zyrtare**. 100% = të gjitha kriteret e detyrueshme `PASS`, jo 100% e rreshtave të kodit.

## 6. Prioriteti inxhinierik i detyrave të mbetura

| Prioritet | Detyrë | Pse paraprin të tjerat | Evidenca për mbyllje |
|---|---|---|---|
| P0 | Klasifikimi DRONE/BIRD/UNKNOWN në video reale | Pa klasifikim nuk funksionon zinxhiri tjetër | Holdout confusion matrix + examples |
| P0 | PostgreSQL ↔ backend ↔ gateway ↔ UI me demo të etiketuar | Dallon bug API nga bug ML | Test real HTTP + screenshot UI |
| P0 | Regresioni UNKNOWN pa ID/trajectory | Kriter dhe siguri semantike | Test negativ DB, API, UI |
| P1 | Persistenca dhe ID tracking multi-object | Tracking vazhdon nga detektori | IDF1/HOTA, switches |
| P1 | Real-map + tactical synchronization | Kriter UI/gjeospatial | Identitetet dhe kohët identike |
| P1 | Range/uncertainty e bazuar në GT | Mos deklaro pozicione pa prova | MAE/RMSE + calibration report |
| P1 | Matje performance dhe kostos | Pragu FPS/latency/cost në sfidë | Hardware baseline + logs |
| P1 | Black-box, tests të pavarura dhe reproducible deploy | Juria e ekzekuton pa ekipin | Fresh clone + CLI + artifacts |
| P2 → në F45 | USB + dron real | Vetëm atëherë physical verification | Field protocol dhe raw evidence |

## 7. Kufizimet aktuale të sigurisë dhe operimit

- Në logun e fundit hapësira e lirë C: ishte **rreth 2.98 GB**; mos fillo build-e/dataset copies të mëdha pa audit disk. Mos përdor `docker system prune`, `down -v`, WSL unregister apo reset.
- **Mos publiko** `deployment.env`, hash-et operator, çelësat, backup-et, volume PostgreSQL, checkpoint-et e modelit ose media pa licencë.
- Operator gateway aktual është loopback-only HTTP. Nëse duhet qasje nga rrjeti, TLS dhe auth multi-user kërkojnë projektim të ri.
- Modelet AI janë artefakte të jashtme të versionuara me SHA, jo të domosdoshme në repository Git normal. Dokumento storage, licencën dhe procedurën e shkarkimit.
- Demo records kanë emrin `[DEMO]`, `source_kind=SIMULATION`, `validation_level=SOFTWARE_ONLY`. Mos i përfshi në raportet e saktësisë së modelit.
- `deploy/compose.local-compat.yaml` dhe credential env duhet të përdoren së bashku në deploy; mos i rikrijo shërbimet nga Compose origjinal pa kontrolluar imazhin PostgreSQL dhe publikimin e portës.

## 8. Komandat minimale për kontroll pa shkatërrim

```powershell
cd C:\Users\nistr\skyguard-fusion
git branch --show-current
git status --short
docker ps --filter 'name=skyguard-fusion' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
Invoke-RestMethod 'http://127.0.0.1:8088/healthz'
# Kontroll API vetëm përmes sesionit operator; ose përdor runner-in E2E.
```

## 9. Paketa e finalizimit V6

1. Publikuesi V6 synon branch-in `feature/skyguard-mvp`, kontrollon hash-in GitHub dhe ndalon në divergence. Ai **nuk publikon** përmbajtjen e panjohur të 4 skedarëve lokalë; vendos në Git template/README/manifest të gjeneruar nga kjo paketë. Lokalisht mund të shfaqen si `modified` për shkak të versioneve personale të ruajtura.
2. Publikohet ky raport, seed-i synthetic dhe runner-i E2E si source; **nuk ekzekutohet automatikisht një seed i databazës nga `git push`**.
3. Testi fizik i kamerës/dronit, trajnimet e tjera dhe threshold-et zyrtare mbeten punë e ardhshme me raporte të ndara.

**Përfundim inxhinierik:** infrastruktura dhe një pjesë e madhe e source-code janë realizuar; mund të demonstrohet persistenca në mënyrë transparente me fixtures. Përmbushja `100%` e sfidës mbetet e pakonfirmuar derisa të ketë metrika të pavarura mbi detektorin, tracker-in, gjeospatial-in, performancën dhe provën fizike të Fazës 45.
