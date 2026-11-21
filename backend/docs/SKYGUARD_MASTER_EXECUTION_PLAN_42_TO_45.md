# SKYGUARD FUSION â€” PLANI I VETÃ‹M EKZEKUTIV (Fazat 42â€“45)

**Versioni:** 1.0 â€” 20.11.2026  
**Objektivi:** pÃ«rmbushje objektive e tÃ« gjitha kritereve tÃ« sfidÃ«s *Automated Drone Detection & Situational Awareness*, jo maksimalizim kozmetik i pÃ«rqindjeve.  
**Politika:** degÃ« `feature/skyguard-mvp`; ndryshime vetÃ«m me PowerShell/terminal; **asnjÃ« git add/commit/push/merge/cherry-pick pa autorizim**.  
**Prova fizike:** kamera USB dhe droni real vetÃ«m nÃ« **FazÃ«n 45** nÃ« kushte tÃ« kontrolluara dhe tÃ« sigurta. Fazat 42â€“44: zhvillim dhe validim softuerik me imazhe/video tÃ« licencuara.

## 0. SKICA FINALE E SISTEMIT â€” NJÃ‹ PIPELINE, DY HARTA

```text
 VIDEO E REGJISTRUAR  (42â€“44)            USB CAMERA + DRON REAL (45)
                 \                            /
                  \---- Ingest + frame/session/timestamp ---->
                        Colab V1 detection + class/UNKNOWN
                                      |
                     Multi-object tracking / identity
                                      |
                        Flight paths / evidence store
                                      |
                  Camera calibration + range evidence (42.6B)
                                      |
        Camera pose ENU + verified facility WGS84 (42.6C)
                                      |
                  ONE VERSIONED LIVE STATE / API CONTRACT
                              /                       \
             REAL GEO MAP                            SKYGUARD TACTICAL MAP
         licensed basemap + WGS84               ENU + bearings + zones + paths
                              \                       /
                             synchronized operator UI
                                      |
                Alerts + security + telemetry + audit/export
```

**TÃ« dy hartat pÃ«rdorin tÃ« njÃ«jtin track_id, session_id, frame_id, timestamp dhe gjendje uncertainty.** Nuk krijohet koordinatÃ« reale/GPS nga njÃ« bearing i vetÃ«m. Skenari `UNKNOWN / bearing_only` Ã«shtÃ« funksionalitet i saktÃ« dhe i domosdoshÃ«m, jo dÃ«shtim. Burimi i bazÃ«s gjeografike duhet tÃ« jetÃ« i licencuar, me atribim, cache dhe mÃ«nyrÃ« offline tÃ« pÃ«rcaktuar. ETH multi-view pÃ«rdoret si **dataset i mundshÃ«m validimi** pÃ«r kalibrim/GNSS, jo si burim hartash, vetÃ«m pasi tÃ« kontrollohen licencat dhe gjurmueshmÃ«ria.

## 1. GJENDJA E KONFIRMUAR DERI SOT

| Komponenti | Evidenca | Vendimi aktual |
|---|---|---|
| Fazat 38 | kod pÃ«r tracking/frame/ID, regresione tÃ« mÃ«parshme | bazÃ« softuerike ekzistuese; kufizime reale mbeten |
| Faza 39 | module kontratash kamera/geo/kalibrim dhe teste | software i avancuar; **distanca reale e pamatur** |
| Faza 40 | skenÃ« harte, geo panel, history/trajectory | software i ndÃ«rtuar; geolocation real i paverifikuar |
| Faza 41 | advisory engine/API/UI, auditim i kaluar | software i ndÃ«rtuar; rreziku i auth dhe alarmet reale mbeten |
| 42.1â€“42.4 | offline MOT, real-video adapter, crossing stress, camera profiler | **878 backend / 56 frontend**, lint/build passed (raport i mÃ«parshÃ«m) |
| 42.5/42.5B | smoke `best.pt` mbi 1 dhe 8 imazhe, inferencÃ« CPU | detektimi punon; **jo provÃ« saktÃ«sie/kapaciteti live** |
| 42.5C | 7,337 train; 861 val; 438 test; test hash duplikatesh=0; benchmark smoke | 438 test: P=.6124, R=.4215, mAP50=.4657, mAP50-95=.2046 **vetÃ«m smoke** |
| Colab V1 (historik) | YOLO11n, 438 test, P=.9345, R=.8316, mAP50=.8625 | rezultat i raportuar; kÃ«rkohet verifikim hash dhe integrim production |
| 42.6A | skript vetÃ«m-lexim pÃ«r checkpoint/video/kontrata i pÃ«rgatitur | **asnjÃ« log ekzekutimi i konfirmuar** |
| 42.6B | `range_evidence_guard_v1.py` + 17 teste tÃ« reja | **895/895 backend** pas instalimit; vetÃ«m matematikÃ«, jo distanca reale |
| 42.6C | kontrata ENU/WGS84 me dy harta | **40 teste lokale kaluan nÃ« mjedisin e pÃ«rgatitjes**; Windows/repo pending |

Rruga e raportuar e modelit kryesor Ã«shtÃ« `backend/models/skyguard-drone-bird-t4-v1/best.pt`, SHA256 referencÃ«: `eee9ea5271fa0e3505e3ff103d4e51202f234445e4c19c3482fa7aa00a122901`. Checkpoint-i `backend/runs/skyguard/drone_bird_smoke/weights/best.pt` **nuk e zÃ«vendÃ«son** modelin kryesor.

**PÃ«rqindjet e mÃ«parshme ishin vlerÃ«sime, jo pÃ«rfundime automatike.** Progresi ndryshon vetÃ«m pasi tÃ« ekzistojÃ« raport pranimi. Kriteret pa testim nÃ« fushÃ« mbeten `FIELD_PENDING`.

## 2. RREGULLA TEKNIKE TÃ‹ PANDRYSHUESHME

1. **Nuk pÃ«rsÃ«risim fazat e kryera.** Ã‡do hap ka ID tÃ« pandryshueshme, hyrje/dalje tÃ« qarta dhe raporte.
2. **NjÃ« pipeline production.** Adapteri offline nuk quhet production tracker, nÃ«se ende nuk Ã«shtÃ« lidhur.
3. **NjÃ« burim identiteti/kohor.** Frame, tracks, paths, advisories dhe harta ndajnÃ« `session_id`, `frame_id`, `frame_index`, `timestamp_ms`, `track_id`.
4. **Metoda shkencore.** Train/val/test ndahen pa rrjedhje tÃ« tÃ« dhÃ«nave; matje nÃ« video reale me etiketa/ID reference kur llogarisim saktÃ«si.
5. **SaktÃ«sia e range:** `Z=f*S/p` me madhÃ«si tÃ« njohur e orientim tÃ« pÃ«rshtatur dhe interval gabimi; kjo **nuk** Ã«shtÃ« GPS/facility distance.
6. **Gjeolokalizimi:** vetÃ«m me pozÃ« kamere, orientim, kalibrim dhe referencÃ« WGS84 tÃ« verifikuara. Pa prova: `UNKNOWN`, bearing, ose relative ENU sipas tÃ« dhÃ«nave.
7. **NjÃ« pikÃ« nÃ« dy harta:** pozicioni WGS84 dhe ENU rrjedhin nga i njÃ«jti vlerÃ«sim; UI nuk shpik coordinates nga pamja vizuale.
8. **Zero sekrete nÃ« frontend.** API key nuk hardkodohet nÃ« bundle publik; backend/session auth duhet tÃ« vÃ«rtetohet me prova negative.
9. **Testim i pavarur:** pa outpute demo tÃ« hardkoduara; inpute negative, kapacitete, recalim, stres dhe edge cases.
10. **Audit:** ruajmÃ« manifestin e checkpoint, versionet, hash-et, licencat, komandat, konfigurimin e harduerit dhe metrikat.
11. **Siguri:** pa ndÃ«rhyrje ndaj dronit; sistemi Ã«shtÃ« vetÃ«m monitorim dhe alarmim.

## 3. NJÃ‹ PLAN I PANDRYSHUESHÃ‹M EKZEKUTIMI

| ID | PÃ«rmbajtja | VarÃ«sia | Kushti konkret i pranimit | Evidenca/statusi |
|---|---|---|---|---|
| **42.6A** | Auditim read-only i checkpoint-it Colab, videove, tracker-it, API dhe frontend | 42.5C | vÃ«rtetohet hash V1; inventari i videove/licencave dhe kontratat e prodhimit | **SKRIPTI GATI / RAPORT PENDING** |
| **42.6B** | Guard matematikor i distancÃ«s nga madhÃ«sia e objektit | 39 | UNKNOWN kur s'ka prova; 17 teste dhe regresion total | **DONE SOFTWARE (895 passed)** |
| **42.6C** | Adapter i vetÃ«m ENU + WGS84 pÃ«r dy harta, me evidencÃ« e uncertainty | 42.6B | track/frame sync, bearing_only, ENU, WGS84 me reference, teste negative | **KOD I PÃ‹RGATITUR / 40 TESTE LOKALE; WINDOWS PENDING** |
| **42.6D** | Integrim i 42.6B/C nÃ« backend ekzistues dhe API, pa duplicim route-sh | 42.6A, 42.6C | contract tests, UNKNOWN, stale frame rejection, geo serialization | **TODO** |
| **42.7A** | Integrim video tÃ« regjistruar â†’ Colab V1 â†’ tracker *production* | 42.6A | real predictions/tracks per frame, identitete, timestamp, prova video, test regression | **TODO** |
| **42.7B** | Benchmark multi-drone me track ID reference / klasifikim & raste tÃ« panjohura | 42.7A | metrika detektimi, IDF1/HOTA/ID switches/fragmentation, video splits tÃ« dokumentuara | **TODO** |
| **42.8A** | Harta gjeografike reale me MapLibre dhe basemap tÃ« licencuar | 42.6D | attribution, fallback, WGS84 or no point, kontroll i network/offline | **TODO** |
| **42.8B** | Harta taktike SkyGuard nÃ« ENU, zonat, histories, sinkronizimi i tÃ« dyjave | 42.7A,42.8A | tÃ« njÃ«jtat ID/frame, trajectory, uncertainty dhe operator controls | **TODO** |
| **42.9** | Auditim E2E i rrjedhÃ«s sÃ« regjistruar Video â†’ AI â†’ Tracks â†’ Geo â†’ API â†’ Dy Harta | tÃ« gjitha 42.x | raporte reproducible, negative/out-of-order/stale tests, pa pretendime hardware | **TODO** |
| **43.1** | Profilim i pipeline-it tÃ« vÃ«rtetÃ« (CPU/GPU), baseline media | 42.9 | FPS, latency P50/P95/P99, RAM/CPU, warm/cold start, drop definition | **TODO** |
| **43.2** | Optimizo bottleneck-et (batch/async/backpressure, ONNX/OpenVINO/GPU kur justifikohet) | 43.1 | benchmark i njÃ«jtÃ« para/pas, pa rÃ«nie tÃ« papranueshme tÃ« accuracy/ID | **TODO** |
| **43.3** | Kapacitet tracking, harduer low-cost, BOM | 43.2 | maksimumi i vÃ«zhguar multi-drone, konfigurimi, versionet, Ã§mimet me datÃ« | **TODO** |
| **44.1** | Security hardening, input limits, RTSP SSRF/rate, RBAC, secrets | 42.9 | negative/abuse tests, pa key nÃ« bundle, threat model | **TODO** |
| **44.2** | Robustness: kamera/video disconnect, corrupt input, restart, race, staleness | 44.1 | fail-closed, restart consistency, telemetry, data integrity | **TODO** |
| **44.3** | Persistence/export/audit + black-box install and reproducibility | 44.2 | ruajtje gjendjeje sipas politikÃ«s, setup clean, unknown inputs, audit final software | **TODO** |
| **45.1** | Validimi fizik me kamerÃ« USB tÃ« vÃ«rtetÃ« | 44.3 | measured FPS/latency, object detection/tracks me konfigurim real | **FIELD RESERVED** |
| **45.2** | Validimi fizik me dron real nÃ« terren tÃ« sigurt | 45.1 | ground truth i pavarur: distanca, pozicion, small/far targets, ID crossing | **FIELD RESERVED** |
| **45.3** | Final demo & release | 45.2 | dy harta live, numri i dronÃ«ve, pozicionet/path, metrika, kosto, install clean | **FIELD RESERVED** |

**Portat e kalimit:**

- **Gate 42:** rrjedhÃ« E2E me video tÃ« regjistruara, drejtim/pozicion me etiketa tÃ« sakta evidencash, dy harta funksionale; pa prova fizike.
- **Gate 43:** metrika throughput/latency/kapacitet/kosto me setup tÃ« riprodhueshÃ«m.
- **Gate 44:** negative/robustness/black-box test suite, hardening dhe dokumentim i plotÃ«.
- **Gate 45:** demonstrim fizik, gabimet e pozicionit/distancÃ«s kundrejt referencave tÃ« pavarura dhe evidenca pÃ«r Ã§do kriter tÃ« sfidÃ«s.

## 4. KRITERET E JURISÃ‹ â†’ MATJA E PRANIMIT

| Kriteri i sfidÃ«s | Metrikat minimale qÃ« DO TÃ‹ RAPORTOJMÃ‹ (jo pragje zyrtare) | Pika e raportimit |
|---|---|---|
| Detection | Precision/recall, mAP50, mAP50-95, FN, false alarms sipas kushteve/madhÃ«sisÃ« | 42.7B / 45 |
| Drone / bird / other | Confusion matrix, unknown handling, FP birdâ†’drone, negative clips | 42.7B / 45 |
| Multi-object tracking | simultaneous count, IDF1/HOTA, ID switches, fragmentation, crossings | 42.7B / 43 / 45 |
| Location & distance | gabim nÃ« metra, MAE/RMSE, error P95, coverage UNKNOWN, pasiguri dhe metoda | 42.6C / 45 |
| Flight paths | track IDs, frame index, sample count, time monotonicity, persistence, trajectory error | 42.8B / 45 |
| Two maps | sync latency, correctness of WGS84/ENU, refresh, UI accessibility, basemap outage | 42.8 / 44 / 45 |
| Real time | input/output FPS, P50/P95/P99, CPU/GPU/RAM, warm start, drop definition | 43 / 45 |
| Low cost | hardware BOM, dates/sources/prices, performance-per-euro and limits | 43.3 / 45 |
| Jury robustness | clean setup, API contract, unknown videos, recovery, no hardcoded responses | 44.3 / 45 |

**OrganizatorÃ«t nuk specifikojnÃ« FPS minimal, kufi distancash apo njÃ« numÃ«r absolut dronÃ«sh.** Nuk shpallim pragje tÃ« shpikura si kushte zyrtare; objektivat e brendshme pÃ«rcaktohen vetÃ«m pas matjeve baseline.

## 5. BAZAT E HARTÃ‹S DHE LICENCAT

- **Harta reale:** MapLibre GL JS + basemap i autorizuar/OSM-compatible me atribim tÃ« dukshÃ«m dhe respektim tÃ« tile policy. OSM nuk Ã«shtÃ« automatikisht njÃ« server tiles falas pÃ«r pÃ«rdorim tÃ« rÃ«ndÃ« komercial; ofruesi final zgjidhet me licencÃ«/terms tÃ« verifikuara.
- **Harta taktike:** rendering i pavarur nga rrjeti nÃ« metra ENU me bearing-only, uncertainty, ID, trajektore dhe zonat.
- **Burimi i gjendjes:** vetÃ«m njÃ« payload me dy projeksione; `.real_map.target_geojson` mbetet `null` nÃ« mungesÃ« tÃ« referencÃ«s valide.
- **ETH Multi-view:** kandidat pÃ«r geo ground truth dhe kamera tÃ« kalibruara; pÃ«rdoret pasi tÃ« verifikohen licenca, formatet dhe konsistenca kohore.

## 6. VENDIMI OPERACIONAL I RADHÃ‹S

1. Ekzekuto 42.6C (new-only, **jo** integrim production) dhe kontrollo raportin 40+895 teste; nÃ«se ka regresione, ndal dhe diagnostiko pa ndryshuar histori Git.
2. Ekzekuto 42.6A (read-only) nÃ«se ende s'ka raport; vetÃ«m pas saj shohim signature tÃ« vÃ«rteta tÃ« prodhimit.
3. BÃ«jmÃ« **42.6D** mbi kontratat e zbuluara, pastaj 42.7A dhe vijojmÃ« tabelÃ«n pa ndryshuar rendin.
4. Ã‡do raport qÃ« dÃ«rgohet pÃ«rditÃ«son kÃ«tÃ« plan: `DONE_SOFTWARE`, `VALIDATED_RECORDED_MEDIA`, `FIELD_PENDING` ose `BLOCKED`.

> **Deklarata finale pÃ«r jurinÃ«, vetÃ«m pas Gate 45:** â€œDronÃ«t u zbuluan, u klasifikuan, u ndoqÃ«n individualisht, iu vlerÃ«suan pozicionet dhe distancat dhe u shfaqÃ«n nÃ« tÃ« dy hartat me gabime dhe kosto tÃ« dokumentuara.â€ PÃ«rpara Gate 45 kjo Ã«shtÃ« objektiv, jo pretendim i realizuar.