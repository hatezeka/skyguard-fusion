# SkyGuard FUSION — deployment contract

The single PowerShell installer places these files, patches the existing Operator Console for **server-session mode only in deployment builds**, and generates credentials **outside the Git repository** under `%LOCALAPPDATA%/SkyGuardDeployment/`.

## Architecture

Browser → Caddy (TLS in domain mode) → session gateway (HTTP-only signed cookie, origin checks, operator allowlist) → existing FastAPI `/api/persistence` (server-injected bearer) → PostgreSQL 17. Alembic migration runs once as a Compose service before backend starts.

**The existing React frontend is used**, not replaced. The UI *reads* recorded scientific evidence and benchmarks; registration/ingestion endpoints are deliberately blocked at the browser gateway. Evidence generation stays in the protected backend. UNKNOWN is not tracked.

## Security boundaries and limitations

- No `VITE_*` API key or frontend service token. `VITE_SKYGUARD_DEPLOYED=true` is only a boolean mode flag. Backend keys live only in a local env file and Docker service environments, not in source control.
- Single operator login with PBKDF2-SHA256 and signed HttpOnly, SameSite=Strict cookies. This is a staging-capable gateway, **not multi-user RBAC / OIDC / enterprise SSO**. Do not claim production security certification from this alone.
- The gateway only permits GET requests to `/api/persistence/*`, plus operator write actions for registered cameras, alert acknowledgement, and live start/stop. Browser cannot POST frames, evidence, or scientific benchmarks; such writes remain in backend.
- Only Caddy exposes a host port. PostgreSQL and FastAPI are accessible only inside private Docker networking. Internet-facing mode needs a real DNS name pointing to the server, inbound ports 80 and 443, and valid TLS from Caddy.
- PostgreSQL volume is persistent. `docker compose down` without `-v` does not delete it. **Never use `down -v`**. Backups, restore drills, retention, IAM review, WAF/rate-limit and monitoring are still deployment acceptance activities.
- Existing local SQLite data is **not** silently copied to PostgreSQL. A new Postgres DB will be empty until backend produces records. Migrate real historical SQLite data only with a separately verified transaction and backup.
- A missing `.pt` file will not be faked; real AI inference still depends on actual model checkpoint and scientific validation. Camera USB passthrough from Windows Docker is **not** assumed to work; field USB/drone test remains Phase 45 native/hardware-specific.
- Production CORS should be restricted; gateway uses same-origin. Avoid publicly exposing the FastAPI listener.

## Commands

```
# Installer: -DryRun, then -Start
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\SkyGuard_FUSION_Deploy_FullStack_OneCommand.ps1" -Repository "C:\Users\nistr\skyguard-fusion" -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\SkyGuard_FUSION_Deploy_FullStack_OneCommand.ps1" -Repository "C:\Users\nistr\skyguard-fusion" -Start
```

For a public deployment, use `-Domain skyguard.example.org -Start` after DNS and TLS prerequisites are met. Without `-Domain`, the installer binds only to `http://127.0.0.1:8088/operator.html` to avoid unintended public exposure.

For backups use a secure PostgreSQL `pg_dump` custom-format backup, verify restore separately, and store encrypted off-host. The installer does not transfer SQLite content or run destructive database resets.

## Media AI input module (deployment build)

The **existing** Operator Console's Monitor page gains a source-media analysis panel. Its browser requests go to `/api/media/*`; an authenticated Python gateway forwards them to the backend using a **server-only** bearer key. The browser may upload JPEG/PNG (<=12MB) or MP4/AVI/WebM (<=40MB), view a PostgreSQL-backed processing job, inspect true AI detections and confirmed DRONE/BIRD track histories in **pixel coordinates**, and preview a backend-annotated JPEG. At most 120 video frames are analyzed per upload by default. File bytes are stored in the isolated persistent `media_data` Docker volume. Review privacy, storage limits and retention before deployment.

The optional laptop camera uses browser `getUserMedia` (HTTPS or localhost) to submit bounded JPEG frames at up to 2 FPS to the backend for real YOLO+tracking+PostgreSQL ingestion; measured FPS/latency can be lower. **This is not a 30-FPS field demonstration.** A private LAN IP camera uses the existing authenticated `POST /api/persistence/live/start` RTSP bridge and is only reachable when Docker's `camera_egress` network can route to the IP address. No public cloud metadata URLs or DNS hostnames are allowed as RTSP targets from this gateway.

Schema revision `0002` introduces durable media-job statuses; the migration is run by **Alembic**, not Prisma. Jobs RUNNING/QUEUED after backend restart become INTERRUPTED (no fabricated completion); video source files remain on the Docker volume for investigation. Scientific verification records / ground truth benchmarks remain **backend-only**. Users cannot submit directly to `/api/persistence/frames`, `/api/persistence/benchmarks`, or `/api/persistence/evidence`.

`SKYGUARD_EXPECTED_MODEL_SHA256` must be the independently verified SHA-256 of the real trained checkpoint. Without both this digest and the physical `.pt` file at `backend/models/skyguard-drone-bird-t4-v1/best.pt`, the web app may deploy and show data, but media analysis **returns MODEL_MISSING/503**. No fallback to generic YOLO weights. The model is **not** bundled into this installer.

Camera GPS alone cannot estimate range to a distant drone. Frontend shows no `range_m`, drone GPS, 3D localization, accuracy precision/recall or validated IDF1/HOTA unless a separately verified, calibrated, ground-truthed source establishes them. A single photo has no trajectory. UNKNOWN has no ID or trajectory. Existing real map works as a basemap with no fabricated drone pins.

This installer is a staging/deployment scaffold requiring live Docker validation, Caddy TLS/DNS verification, signed release/reproducibility, backups and restore drill, media retention policy, independent security audit, and real-camera/real-drone Phase 45 testing before a production certification claim. Public deployments should use strong multi-user identities/RBAC, stronger perimeter protection and monitoring rather than the single-operator password alone.
