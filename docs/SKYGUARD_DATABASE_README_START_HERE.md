# SkyGuard FUSION — Database Implementation Package

**Additive package** — does not overwrite existing backend/frontend files or make Git commits.

**Current verification:** 22 isolated persistence tests with SQLite; PostgreSQL service and integration into the actual SkyGuard live frame callback are not validated yet.

1. Read `docs/SKYGUARD_DATABASE.md`.
2. On Windows extract `SkyGuard_FUSION_Database_Full.zip` into a **new temp folder**.
3. Run `scripts/install_skyguard_database.ps1 -Repository C:\Users\nistr\skyguard-fusion` from that folder.
4. Install Python requirements, run Alembic migration, run tests.
5. Use the composed FastAPI runner to mount this router without changing existing source files, after checking your actual backend FastAPI entrypoint.

The database records supplied AI results, never invents classifications, GPS, range or performance. UNKNOWN receives no tracking ID/trajectory. Real USB camera/drone validation is reserved for Phase 45.

No commit, push or merge is performed by this package.
