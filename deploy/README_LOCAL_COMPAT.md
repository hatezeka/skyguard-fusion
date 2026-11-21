# SkyGuard FUSION - validated local Docker compatibility

This is the **local, loopback-only** configuration validated on 2026-11-21.
It addresses `exec format error` for the Alpine PostgreSQL/Caddy containers on the
specific Docker Desktop installation while preserving PostgreSQL 17, real DB
migrations, CPU-only PyTorch, and the gateway UI. It is **not** a public HTTPS setup.

## Prerequisites

- Docker Desktop running Linux containers (`linux/amd64` or `linux/x86_64`).
- Existing Docker images: `skyguard-fusion-backend:latest`, `skyguard-fusion-migration:latest`,
  `skyguard-fusion-gateway:latest` and official `postgres:17-bookworm`.
- A local secret environment file at `%LOCALAPPDATA%\SkyGuardDeployment\deployment.env`.
  **Never commit or share this file**, its backups, passwords, hashes or signing keys.
- Existing named PostgreSQL volume must be preserved. Never use `down -v`.

## Restart the local stack without rebuilding or downloading

From the repository root in Windows PowerShell:

```powershell
$secrets = Join-Path $env:LOCALAPPDATA 'SkyGuardDeployment\deployment.env'
$files = @('--env-file', $secrets,
  '-f', '.\deploy\compose.yaml',
  '-f', '.\deploy\compose.local-compat.yaml')

docker compose @files config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Invalid Compose configuration' }

docker compose @files up -d --no-build --pull never postgres migration backend gateway
if ($LASTEXITCODE -ne 0) { throw 'SkyGuard service startup failed' }

docker compose @files ps
Invoke-RestMethod 'http://127.0.0.1:8088/healthz'
```

Open `http://127.0.0.1:8088/login.html` using the locally configured `operator`
account. Set or reset its password through a secure local workflow; no password
is stored in this repository. Repeated failed logins may trigger rate limits.

The `https-edge` profile prevents starting the incompatible Alpine Caddy image in
this local mode, and prevents a host-port conflict on 8088. The base Compose
file remains available for a separately validated HTTPS deployment.

## What this does NOT validate

Passing deployment, authentication and Alembic `0002` does not establish real
DRONE/BIRD/UNKNOWN classification accuracy, tracking persistence, FPS/latency,
range/GPS estimation, or physical USB-camera/drone field test results. Those
require separate recorded scientific evidence. `UNKNOWN` must not receive a
tracked target ID or trajectory.

Do not run Docker prune/reset or delete PostgreSQL named volumes to restart.
The CPU-only backend Dockerfile has its own prior Git commit (`e12abc5`).
