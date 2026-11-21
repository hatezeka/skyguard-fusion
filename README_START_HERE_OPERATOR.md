# SkyGuard Operator Console — Authenticated Local Deployment

**Login:** `http://127.0.0.1:8088/login.html` · user `operator` · private locally configured password. The session cookie is HttpOnly and gateway authentication is distinct from the database's service-to-service key.

**Console:** `http://127.0.0.1:8088/operator.html`.

**Backend persistence demo:** After installing the kit, run `scripts/Run_SkyGuard_Demo_Seed.ps1 -Seed -VerifyGateway`, then reload the console and select the `SIMULATION` session. This fixture contains 15 synthetic detections / 2 tracks / 2 alerts over 5 frames. UNKNOWN never has a track. Geospatial, FPS and latency remain blank because no scientific ground truth or measured timing is provided.

**Important:** The source React app and built gateway asset are not interchangeable. If `/api/persistence` returns demo rows but the UI displays `DISCONNECTED`, inspect the frontend's gateway-cookie integration and rebuild only the gateway when disk capacity permits. **Never expose `SKYGUARD_DB_API_KEY` to the browser or Vite environment.**

See `docs/SKYGUARD_TECHNICAL_STATUS_ROADMAP_AND_ACCEPTANCE.md` for engineering acceptance, remaining roadmap and evidence required for Phase 45.
