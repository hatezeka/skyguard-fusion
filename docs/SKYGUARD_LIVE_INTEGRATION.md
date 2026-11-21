# SkyGuard FUSION — Live → SQLite → FastAPI → Operator Console

## Scope / verified limitations

This patch does not train a neural net and does not assert field validation. It does NOT infer range, GPS, or real target positions from monocular video. In the original camera source there is no hardware capture timestamp, so persisted `captured_at` is **host UTC frame-publication time**, not camera sensor timestamp. No raw video is archived; the SHA-256 hash, when present, refers to the transient server-annotated JPEG. Physical USB/drone validation remains for Phase 45.

### Flow

1. The existing `LiveSessionManager._worker()` produces a real detector/tracker snapshot for each decoded frame. One short callback is added **after** the snapshot is published, outside the manager lock.
2. The callback copies original detection + tracker object associations into a bounded in-memory queue; UNKNOWN is never assigned identity or geographic information.
3. A single database writer creates a camera record and a `LIVE / SOFTWARE_ONLY` session, then persists **frame + detections + confirmed track observations** transactionally. Latency records use existing `processing_seconds`; no FPS values are fabricated.
4. All existing persistence endpoints continue at `/api/persistence/*`; `GET /api/persistence/live/frame` returns a **synchronized JPEG and its exact-frame metadata** with Bearer auth; `/api/persistence/live/health` exposes explicit dropped-frame/error counters.
5. The React operator console accesses persistence + live frame through the Vite localhost proxy using the same in-memory local key. The original camera `/api/v1/live/*` routes retain their existing separate `X-API-Key` check; launcher sets both keys in the private backend process environment.

The camera is not automatically started by this patch. Starting a camera requires an available YOLO checkpoint and a real video source. The Operator Console offers explicitly authorized USB camera Start/Stop buttons under Live Observation; its bridge routes call the **existing** start/stop validation and model-check logic. These buttons are intended only for the later controlled Phase 45 physical validation, not as proof of passing it. Software-only video benchmark commands remain separate. Live state / records are never simulated.

### Security and acceptance

- Run only on `127.0.0.1` with one uvicorn worker and without `--reload` (the recorder is per-process).
- Protect API credentials. The local service key is printed once and held only in memory. Do not put it in a Vite environment variable, git, URLs, localStorage, or screenshots.
- The queue has a finite capacity (2048 frames). If it overflows, `lost_frames` and `last_error` report degradation. Never present these recordings as complete when this indicator is nonzero.
- Camera metadata is created without GPS or calibration assumptions. UNKNOWN detections are archived as classifications only.
- Existing camera, tracker and science algorithms are not modified. The sole inserted source block observes real `detections` and `tracks` emitted by the engine.
- The bridge is intended for **local evaluation** with the installed SQLite schema (revision 0001) and has not been field-benchmarked.
- Tests use explicitly synthetic **test fixtures** only, and do not verify model accuracy or camera performance.

### Acceptance after local install

```powershell
cd C:\Users\nistr\skyguard-fusion\backend
& .\.venv\Scripts\python.exe -m pytest -q tests\test_skyguard_live_integration.py tests\test_skyguard_persistence.py tests\test_phase43_45_scientific_contract.py tests\test_phase43_45_inventory.py
cd ..\frontend
npm.cmd exec -- vitest run src/operator/logic.test.ts
npm.cmd run build -- --config operator.vite.config.ts
```

Then launch via `scripts\start_skyguard_integrated.ps1`, open `http://127.0.0.1:5173/operator.html`, select Integration Settings and enter printed local key. The original interface remains at `/`, but its existing X-API-Key authorization is unchanged. Use the Operator Console Start/Stop buttons for authenticated local USB testing when Phase 45 begins. Once actual frames are produced, verify session selection, detections, DRONE/BIRD-only identity history, and recorder health. Restart services and check that the DB still contains the same records. The hardware/detection acceptance criteria require separate Phase 45 evidence.

### Worktree policy

All changes are local to `feature/skyguard-mvp`. Installer creates backups of original files outside the repo. It never commits or pushes. If a code anchor differs, the installer aborts rather than overwriting unknown code. Database data lives under `backend/.skyguard_data/` and must stay ignored.
