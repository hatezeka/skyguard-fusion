# SkyGuard FUSION — 10 backend regression failures (November 21, 2026)

Source: user's diagnostic ZIP; baseline 1026 passed, 10 failed. This change is software-only.

- Four failures stem from git checkout CRLF conversion of *bundled* synthetic fixtures, while pinned SHA-256 values are for LF. The original trusted LF hashes are **unchanged**. The guard permits an exactly matching CRLF-only conversion, but rejects arbitrary/mixed/edited payloads; evidence reports continue to include the original input's raw SHA-256.
- Five advanced tracking failures occur because legacy tests used UNKNOWN as a tracked/ambiguous object. Current binding intentionally excludes UNKNOWN from production tracking. Updated fixtures include a DRONE, BIRD and UNKNOWN, and make ambiguity refer to BIRD, verifying UNKNOWN is still visible as detection, with no track IDs or trajectories. Adds a fail-closed check against noncompliant tracker outputs.
- One trajectories test expected UNKNOWN to have a flight path; this now verifies exactly the opposite, per current explicit project specification.

This is a **contract migration**, not a claim of increased classifier accuracy. It does not enable 3-class V2 weights, fill the actual SQLite DB, or demonstrate video E2E, physical range, GPS, or live FPS. Full backend suite must pass on user's Windows checkout before software acceptance.

The 10 failures have been addressed in source/test contracts; no pass count is asserted until Windows pytest is executed. Field validation remains Phase 45.
