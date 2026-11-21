# SkyGuard FUSION — Operator Console

## Purpose

A complete, responsive, evidence-aware React operator interface added **alongside** the existing SkyGuard frontend. The existing `src/App.tsx`, existing `index.html`, package.json and vite.config.ts are **not overwritten**. Open `/operator.html` in Vite development or build the multi-page release with `operator.vite.config.ts`.

## Architecture and constraints

- React + TypeScript + Vite. Uses the existing React Leaflet and Leaflet dependencies declared in the user's frontend/package.json.
- Persistence uses existing `app.skyguard_persistence.composed:app` and existing `/api/persistence/*` endpoints; no invented backend endpoints.
- 10 sections: command center, observation, identity tracking, dual maps, alerts, evidence, performance, cameras, scientific validation, integration settings.
- UNKNOWN is classification only, never a stored track/trajectory; invalid records are suppressed.
- Both map views use the same persisted tracked detection references and explicit geospatial evidence. The tactical view is computed only from a georeferenced calibrated camera and WGS84 object fixes. Range alone is **not** sufficient for a bearing and is therefore not placed.
- No GPS inferred from a single camera; no fabricated FPS/latency/detections/field-test scores.
- Offline local video preview must not be misrepresented as AI inference. Live status adapter is configurable because its existing backend schema has not yet been inspected; no path is guessed.
- Database query limits remain 200 records per endpoint and 100 for audit/benchmarks. **Not a complete historical export.**
- Restricted local development API key input stored in React memory only; not in VITE_ env, localStorage or URL; direct service key is blocked outside localhost.
- Browser-side service keys are not a production security design. Deploy an authenticated user-session BFF/reverse proxy (e.g., cookie, CSRF, RBAC, TLS) and turn off direct browser bearer credential access for production.
- Frontend key connects to persistence only; the AI live pipeline must separately send valid POST /api/persistence/frames from the backend. No automatic insertion is claimed.

## Install (PowerShell)

1. Download the ZIP into `C:\Users\nistr\Downloads\SkyGuard_FUSION_Operator_Console_Full.zip`.
2. On the `feature/skyguard-mvp` branch:

```powershell
$repo = 'C:\Users\nistr\skyguard-fusion'
$zip = Join-Path $env:USERPROFILE 'Downloads\SkyGuard_FUSION_Operator_Console_Full.zip'
if (-not (Test-Path $zip)) { throw "Missing: $zip" }
$dest = Join-Path $env:TEMP 'SkyGuard_Operator_Install_20261121'
if (Test-Path $dest) { throw "Already extracted: $dest" }
Expand-Archive -LiteralPath $zip -DestinationPath $dest
& "$dest\scripts\install_skyguard_operator.ps1" -Repository $repo -PackageDirectory $dest
```

To start backend and UI later:

```powershell
& 'C:\Users\nistr\skyguard-fusion\scripts\start_skyguard_operator.ps1'
```

If PowerShell blocks local scripts, use `powershell.exe -NoProfile -ExecutionPolicy Bypass -File <script path>` from your own terminal. Do not change machine-wide policy.

## Browser

`http://127.0.0.1:5173/operator.html`. Start script prints a random ephemeral local key; paste it under **Integration settings**. Never publish/screenshot it. The source local SQLite database already exists after successful migrations.

## Caveats

This package was syntactically checked in the packaging environment, but cannot be built against the user's full checkout here because the repository source is not accessible from this environment. The provided installer runs real npm/vitest/build checks in the user's checkout and stops on failure; do not claim it is deployed before those checks pass.

The script does **not** commit, push or delete any original tracked files. Files that conflict with existing differing content cause an error. Do not commit `.skyguard_data/` or secrets.
