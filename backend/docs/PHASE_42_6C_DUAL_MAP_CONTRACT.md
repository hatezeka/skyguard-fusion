# Phase 42.6C â€” Shared scientific projection for two maps (software only)

## Purpose and integration boundary

Create a **single versioned payload** shared by the real geographical map and SkyGuard's own tactical top-down map. No changes to the production `live_map_scene.py`, API, tracker, or UI are made by this phase. Existing route contracts must be audited (42.6A) before the wiring task. The new module is an additive geometry/contract primitive, NOT a second live tracker, live geo service, or production basemap.

## Inputs required

1. Matching session, frame, timestamp, track ID, pixel center and class label.
2. Camera intrinsics (fx, fy, cx, cy, resolution), explicitly flagged calibrated and carrying provenance.
3. Independently surveyed camera position relative to the facility in eastâ€“northâ€“up metres, and an orthonormal proper camera-to-ENU rotation matrix. Optical axes in OpenCV convention: +x right, +y down, +z forward.
4. Optional `RangeAssessment` from `range_evidence_guard_v1.py` (42.6B), which reports **camera optical-axis depth Z**, not slant range, building distance or GPS. `frame_id` and `track_id` must match exactly.
5. Optional verified WGS84 **facility origin** with altitude and survey provenance. Without it, no geographic target point is produced.

## Outputs and safety states

- `unknown`: calibration/pose/observation invalid. Both map payloads null.
- `bearing_only`: calibrated ray direction only; both target positions null; never draw a guessed distance or geographic point. A verified real-map camera marker is allowed when surveyed facility coordinates are supplied.
- `relative_position_only`: local ENU target coordinates exist, but world-map target remains null because no trusted WGS84 reference is available.
- `geographic_position_estimate`: local ENU position and matching WGS84 target GeoJSON point are derived from the **same** observation and depth interval. Both carry identical session/frame/track/time metadata.

`facility_distance_m` is measured from the **facility ENU origin**, not the camera, and is only available in depth-evidenced states. `facility_distance_depth_bounds_m` is a mathematical interval **conditional solely on the bounded input optical depth**, treating intrinsics, camera attitude/position, effective target-size and pixel-center values as exact after upstream evidence checking. It is **not** a confidence interval, nor an estimate of full physical uncertainty. These missing uncertainty sources are explicitly flagged via `uncertainty_scope` and require extension before operational use.

## Geodesy

WGS84 geodetic facility origin (latitude, longitude, altitude) â†’ ECEF â†’ local ENU displacement â†’ target ECEF â†’ WGS84 geodetic. GeoJSON coordinates order is `[longitude, latitude, altitude_metres]` (EPSG:4979). No internet tiles or arbitrary site coordinates are required or constructed in this module.

## Real map and tactical map

The eventual real basemap should be rendered by a licensed provider (for example OSM-compatible tiles) with required attribution and tile usage compliance; MapLibre GL JS may display `target_geojson`. The existing SkyGuard tactical map should use `position_enu_m` from the same payload. Never imply that the basemap itself measures object position. On map tiles/network outage, tactical operations should still work with local ENU and an explicit offline indicator. Never expose private facility coordinates publicly by default.

## Tests / evidence

Targeted `pytest` tests cover numeric transform geometry, GeoJSON coordinate order, ENU+WGS84 synchrony, mismatched frame/track IDs, unverified pose, wrong rotation determinants, pixel out of bounds, unsafe numeric inputs, and fail-closed missing range, plus backend regressions. These tests use controlled mathematical fixtures (not real drone flight data). Actual camera pose calibration, range estimation errors and GNSS accuracy must be evaluated with independently surveyed ground truth in Phase 45. Optional public research datasets require separate license/provenance validation.

## Required future wiring (NOT done by 42.6C)

- 42.6D: inspect actual `api/geo.py` and `live_map_scene.py`, map new payloads into the existing protected API without duplicate routes, add UNKNOWN/stale-frame handling.
- 42.7: connect confirmed Colab V1 inference and recorded-video multi-target tracks using exact session/frame/track IDs, inspect true video licenses and annotation provenance.
- 42.8: integrate two synchronized views, real map tiles/licensing and own tactical map, while preserving operator-friendly bearing-only state.
- 42.9: independent end-to-end recorded-video and negative tests, gate report.
- 45: real USB camera and drone field test with measured distance and errors.

## Invocation and Git safety

A create-only PowerShell script writes precisely this module, test and documentation file, then executes targeted tests and full backend regression. Before writing it captures Git HEAD/index/status, refuses to overwrite existing files, and verifies existing staged/source work remains unaffected. No Git add, commit, push, merge or cherry-pick. Local generated Python `__pycache__` is suppressed through `PYTHONDONTWRITEBYTECODE=1` and pytest cache disabled. Artifacts are saved only in `Downloads`.