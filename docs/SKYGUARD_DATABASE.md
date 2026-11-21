# SkyGuard FUSION — Persisted Evidence / Database v1

## What this package does

Additive FastAPI + SQLAlchemy + Alembic persistence layer. It DOES NOT overwrite the existing video, tracking, map or alert code. It records what the pipeline supplies; **it does not perform AI inference, calculate GPS or infer distance from one camera**. It is ready to be called by the actual pipeline; connecting the existing frame callback requires checking the real backend entry point and payload schema.

- **SQLite**: runs without Docker, for reproducible local development.
- **PostgreSQL**: supported through `postgresql+psycopg://...`, for deployment with multiple operators.
- All writes are transactional. Frames are idempotent on `(session_id, frame_index)` with a content hash. Modified replay returns HTTP 409 rather than silently overwriting results.
- `UNKNOWN` is logged as a detection with no track or geospatial record. `DRONE` / `BIRD` may have persistent tracks and observations if the tracker supplies a stable external ID.
- Coordinates, range and uncertainty need explicitly supplied calibration and provenance; the DB doesn't calculate or validate scientific truth. This is an evidence *input contract*, not proof that coordinates are accurate.
- Evidence-file SHA-256 is computed from real files within an administrator-configured root folder, not accepted on trust from an HTTP caller. No raw videos are stored in PostgreSQL/SQLite.
- Validation status starts `SOFTWARE_ONLY`; a client cannot label a session or benchmark `PHYSICAL_VERIFIED` through these endpoints.
- Both `/api/persistence/*` HTTP GET and POST endpoints **fail closed** unless `SKYGUARD_DB_API_KEY` is set. Do not put this key in Vite `VITE_*` env vars or in a shipped browser bundle. For production, authenticate a backend-for-frontend using real user sessions/RBAC and retain the key server-side.

## Tables (12)

| Table | Contents |
|---|---|
| `sg_cameras` | Camera sources and optional calibrated location |
| `sg_sessions` | Monitoring sessions and evidence validation level |
| `sg_frames` | Unique frame index, capture UTC and payload digest |
| `sg_detections` | DRONE / BIRD / UNKNOWN observations and XYXY bbox |
| `sg_tracks` | Stable DRONE/BIRD IDs scoped to session |
| `sg_track_observations` | Timestamped trajectory associations |
| `sg_geo_evidence` | Explicit geospatial/range estimates with uncertainty and method |
| `sg_alerts` | Alert state and operator acknowledgement |
| `sg_performance_samples` | Measured FPS and latency; method/source |
| `sg_evidence_files` | Validated file path, SHA-256 and size |
| `sg_benchmark_runs` | Dataset hashes, validation level and metrics JSON |
| `sg_audit_events` | Append-only from application API action events (not DB-level WORM) |

## Install (Windows PowerShell; no commit/push)

```powershell
# Unzip the downloaded package in a temporary directory first.
$repo = 'C:\Users\nistr\skyguard-fusion'
$zip  = Join-Path $env:USERPROFILE 'Downloads\SkyGuard_FUSION_Database_Full.zip'
$temp = Join-Path $env:TEMP 'SkyGuard_Database_Install'
if (Test-Path $temp) { throw "Destination already exists: $temp" }
Expand-Archive -LiteralPath $zip -DestinationPath $temp
& (Join-Path $temp 'scripts\install_skyguard_database.ps1') -Repository $repo

Set-Location (Join-Path $repo 'backend')
$python = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { py -m venv .venv }
& $python -m pip install -r requirements.txt
& $python -m pip install -r requirements-persistence.txt
& $python -m alembic -c skyguard_db.ini upgrade head
& $python -m pytest -q tests/test_skyguard_persistence.py
```

**Do not extract directly on top of the project**. The installer checks branch and all destination files before copying. If it reports an existing file, compare changes manually rather than overwriting.

The default SQLite file is `backend/.skyguard_data/skyguard.sqlite3` (gitignored by `backend/.skyguard_data/.gitignore`). Verify it exists after `alembic upgrade head`. Do not commit database files, keys or evidence videos.

### Local standalone API

```powershell
Set-Location 'C:\Users\nistr\skyguard-fusion\backend'
$env:SKYGUARD_DB_API_KEY = (& .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))").Trim()
& .\.venv\Scripts\python.exe -m uvicorn app.skyguard_persistence.standalone:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/docs`; click **Authorize** and enter the generated bearer token to call endpoints. Default CORS permits only localhost Vite development origins. This local standalone mode does **not** include existing AI/streaming endpoints; it is the persistence API only.

### Combined API using the existing FastAPI instance

If the existing instance really is `app.main:app`, run:

```powershell
$env:SKYGUARD_MAIN_APP = 'app.main:app'
& .\.venv\Scripts\python.exe -m uvicorn app.skyguard_persistence.composed:app --host 127.0.0.1 --port 8000
```

If it lives elsewhere, set `SKYGUARD_MAIN_APP` to the existing `module:variable`. The composed runner imports that real instance and adds the `/api/persistence` router without changing the original file. **Confirm the actual entrypoint before using this**, and never run two backend instances on the same port. If the existing app has its own auth/routing, review that integration before exposing it beyond localhost.

To include the router explicitly within an existing FastAPI app, after its creation:

```python
from app.skyguard_persistence import create_router
app.include_router(create_router())
```

Do not run `create_schema_for_tests_only()` in production; Alembic migrations are the source of truth.

## PostgreSQL (optional, requires Docker)

```powershell
Set-Location 'C:\Users\nistr\skyguard-fusion'
$env:SKYGUARD_POSTGRES_PASSWORD = (& .\backend\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))").Trim()
docker compose -f docker-compose.skyguard-db.yml up -d
# The generated URL-safe password can be inserted directly. Use a secure secret manager for deployment; environment variables in PowerShell are temporary and must be set again in a new session.
$env:SKYGUARD_DATABASE_URL = "postgresql+psycopg://skyguard:$($env:SKYGUARD_POSTGRES_PASSWORD)@127.0.0.1:5432/skyguard"
Set-Location backend
& .\.venv\Scripts\python.exe -m pip install -r requirements-persistence-postgres.txt
& .\.venv\Scripts\python.exe -m alembic -c skyguard_db.ini upgrade head
```

For PostgreSQL backups use `pg_dump` in a secured environment, verify restore periodically, and manage credentials with a secret store. PostgreSQL integration is implemented but was not tested against a running PostgreSQL server as part of this package validation.

## Endpoints

All paths have prefix `/api/persistence` and require `Authorization: Bearer <server-issued-token>`.

| Method | Path | Action |
|---|---|---|
| GET | `/health` | Connection check |
| POST/GET | `/cameras` | Register/list cameras |
| POST/GET | `/sessions` | Start/list sessions |
| POST | `/sessions/{id}/close` | Close session |
| GET | `/sessions/{id}/summary` | Observed counts |
| POST | `/frames` | Atomic frame + classifications + tracks + geo |
| GET | `/detections` | Filterable detection history |
| GET | `/tracks` | Stable tracks |
| GET | `/tracks/{id}/history` | Trajectory history |
| GET | `/geospatial` | Evidenced geo/range entries only |
| POST/GET | `/alerts` | Record/list alerts |
| POST | `/alerts/{id}/acknowledge` | Operator acknowledgement |
| POST/GET | `/performance` | Metrics samples |
| POST/GET | `/evidence` | Hash/register files; list metadata |
| POST/GET | `/benchmarks` | Provenance-aware benchmark metrics |
| GET | `/audit` | Recent API action audit entries |

### Call it from the existing pipeline

```python
from app.skyguard_persistence import Database, Store
from app.skyguard_persistence.schemas import CameraCreate, SessionCreate, FrameIngest
store = Store(Database())  # call after Alembic upgrade head
cam = store.create_camera(CameraCreate(name="Ground Camera", source_kind="FILE"))
ses = store.create_session(SessionCreate(camera_id=cam["id"], source_kind="VIDEO"))
result = store.ingest_frame(FrameIngest(
    session_id=ses["id"], frame_index=0, captured_at="2026-11-20T12:00:00Z",
    objects=[{"classification":"UNKNOWN", "bbox":[10,20,30,40]}],
))
```

This is illustrative input, **not synthetic ground-truth, not a test result, and not evidence of real detection**. Feed only actual validated pipeline outputs. The existing pipeline does not automatically persist until its frame processing code is explicitly wired to this Store.

## Two maps and frontend

`frontend/src/api/skyguardPersistence.ts` contains a typed history client. It does not change `App.tsx`. Use `GET /geospatial` to populate a licensed geographic map only when latitude and longitude are not null; use `GET /tracks/{id}/history` for pixel-based tactical trajectories. Do not translate image pixels to GPS without a real calibrated geospatial transform. Keep UNKNOWN out of both track maps. The existing dashboard must be connected explicitly after matching its real data types.

## Safe operational practices / limits

1. Place API behind authenticated operator UI and TLS; the single bearer key is service-level protection, **not** complete RBAC, user attribution or a production security boundary. `acknowledged_by` is caller-supplied and not cryptographically linked to a login. Implement real user auth/RBAC before multi-operator rollout.
2. Rows are durable once committed, but this isn't automatically a tamper-proof or legally compliant audit system. Protect files/DB with filesystem rights, replication, secure backups and retention policy; hash chain/sign audit events if needed.
3. Data requests are capped to 200 rows (500 for per-track history). Implement cursor pagination/aggregation for very large production datasets.
4. Metrics are stored as provided and not independently verified by this module. Final field FPS, latency and range accuracy must be measured in Phase 45 with real USB camera/drone and controlled procedure.
5. Postgres should be load-tested and migration-tested on an actual instance before declaring it production ready.
6. Never infer geo/distance from a single uncalibrated camera; uncertainty and evidence refs must accompany estimates.
7. SQLite backup utility: `& .\scripts\backup_skyguard_sqlite.ps1` from repository root (defaults to Documents\SkyGuardBackups; periodically transfer to secured offline storage). A backup must be restored in a separate environment to verify it.
8. Database access must not block real-time inference: ingest asynchronously with bounded worker queues and observable queue lag in the existing pipeline; this module intentionally does not modify its timing-critical code.
