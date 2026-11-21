# SkyGuard FUSION — Faza 42.6D (Integrimi incremental i dy hartave në API ekzistuese)

## Qëllimi
Përgjigjja e **endpoint-it ekzistues** `POST /api/v1/live/map/scene` përfshin
`dual_map_v1` pa shtuar endpoint të dytë, pa hequr fushat e mëparshme dhe pa
ndryshuar mekanizmin e autorizimit të route-it. Kjo është një fazë integrimi
**kontrate/API**, jo demonstrim i pikave GPS të dronëve.

## Integrimi ekzistues
- `backend/app/localization/live_scene_dual_map_adapter_v1.py`: merr strukturën
  aktuale `live_camera_bearing_map` dhe prodhon dy pamje nga e njëjta skenë.
- `dual_map_v1.tactical_map`: `LOCAL_ENU_METRES_FROM_FACILITY`; për secilën
  detektim regjistron ID vetëm kur ekziston dhe e dallon nga `detection_index`.
- `dual_map_v1.real_map`: `WGS84_EPSG_4979`; lista e vlerësimeve të target-it
  është **bosh** pa evidencë të besueshme të pozicionit.
- `dual_map_v1.legacy_display_geojson`: kopje e GeoJSON të kamerës/bearing
  guide të route-it ekzistues; kjo nuk përmban distancën e dronit.
- Nuk pranohet `verified=true` nga browser-i për të shpikur pozicion.

## Evidenca e ardhshme nga pipeline-i, JO browser-i
`TrustedTrackEvidence` mund të injektohet **vetëm nga një burim server-side**,
kur rrjedha Video→AI→tracking (42.7) ofron `session_id`, `frame_id`,
`frame_index`, `timestamp_ms`, `track_id` dhe detektimin e lidhur. Adapteri
refuzon evidencën me session/frame/time/track/class/confidence të ndryshëm;
kur evidenca kalon, thërret 42.6C `project_dual_maps` dhe përdor guard-in 42.6B.

Route-i aktual **nuk ofron ende** një burim të pavarur server-side për
madhësinë fizike të dronit, pose të verifikuar, GPS të objektit dhe depth me
kufij pasigurie. Kështu ai prodhon vetëm `unverified_direction_only`,
`position_enu_m: null` dhe `target_geojson: null`; ky është rezultati i saktë.

Edhe me koordinata të llogaritura nga të dhëna të deklaruara, validimi në fushë
mbetet i hapur deri në Fazën 45. Intervale të depth nuk janë intervale 95%
të gjeolokalizimit ose gabim total të kalibrimit.

## Testet, kriteret e pranimit dhe varësitë
- Teste adapteri: mungesë range/GPS; ruajtje e përgjigjes legacy; UNKNOWN;
  input invalid; stale/foreign frame evidence; ID mismatch; trajektore të
  paqarta; projeksion ENU/WGS84 me të dhëna sintetike të *etiketuara*.
- Teste API: endpoint-i ekzistues mbetet i regjistruar vetëm një herë;
  thërret adapterin e ri saktësisht një herë.
- Regresioni backend duhet të mbetet pa dështime. Git staging/HEAD dhe
  skedarët e tjerë ekzistues ruhen identikë; skripti rikthen ndryshimet
  e kësaj faze nëse ndonjë provë dështon.
- **Ende e hapur në 42.7**: videoreader + Colab V1 + tracker prodhimi +
  timestamp/track_id të sinkronizuar, pastaj injektimi i evidencës së besuar.
- **Ende e hapur në 42.8**: UI MapLibre me tiles të licencuara, hartë taktike,
  sinkronizim i të dy hartave dhe kontrolle operatori.
- **Ende e hapur në 45**: kamera USB/dron real, pozicioni dhe distanca fizike,
  saktësi dhe gabime të matura, kosto përfundimtare.