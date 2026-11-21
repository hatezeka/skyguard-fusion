# SkyGuard FUSION — Quick Start (2026-11-21)

Repository: `hatezeka/skyguard-fusion` · branch `feature/skyguard-mvp`.

**Full engineering status, evidence limitations and roadmap:** [`docs/SKYGUARD_TECHNICAL_STATUS_ROADMAP_AND_ACCEPTANCE.md`](docs/SKYGUARD_TECHNICAL_STATUS_ROADMAP_AND_ACCEPTANCE.md)

The project has a **loopback-only local Docker deployment**, operator gateway, PostgreSQL, Alembic 0002, backend and frontend. It is NOT a public or certified field deployment. Source and successful deployment are not proof of AI scientific accuracy.

- Login: `http://127.0.0.1:8088/login.html` · username `operator` · use the **private** previously set password (never commit it).
- UI: `http://127.0.0.1:8088/operator.html`.
- DB demo fixture: see `scripts/Run_SkyGuard_Demo_Seed.ps1` (explicit opt-in `-Seed`, then `-VerifyGateway`). Synthetic only; no fake GPS, range, FPS or benchmark figures.
- Postgres compatibility: see `deploy/README_LOCAL_COMPAT.md` and check merged Compose before restarting. Do not run destructive Docker cleanup.
- Never push `deployment.env`, credential hashes, private data volumes, model weights, or unlicensed datasets.

**Do not claim challenge completion** without independent dataset/video and Phase 45 camera/drone verification.
