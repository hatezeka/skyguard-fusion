# SkyGuard Operator Console — Authenticated Local Deployment

**Login:** `http://127.0.0.1:8088/login.html` · user `operator` · private locally configured password. The session cookie is HttpOnly and gateway authentication is distinct from the database's service-to-service key.

**Console:** `http://127.0.0.1:8088/operator.html`.

**Backend persistence demo:** After installing the kit, run `scripts/Run_SkyGuard_Demo_Seed.ps1 -Seed -VerifyGateway`, then reload the console and select the `SIMULATION` session. This fixture contains 15 synthetic detections / 2 tracks / 2 alerts over 5 frames. UNKNOWN never has a track. Geospatial, FPS and latency remain blank because no scientific ground truth or measured timing is provided.

**Important:** The source React app and built gateway asset are not interchangeable. If `/api/persistence` returns demo rows but the UI displays `DISCONNECTED`, inspect the frontend's gateway-cookie integration and rebuild only the gateway when disk capacity permits. **Never expose `SKYGUARD_DB_API_KEY` to the browser or Vite environment.**

See `docs/SKYGUARD_TECHNICAL_STATUS_ROADMAP_AND_ACCEPTANCE.md` for engineering acceptance, remaining roadmap and evidence required for Phase 45.

## USB calibration on Windows

1. In the existing Camera Registry API, register the USB camera with its confirmed local index and stream dimensions. Use the API host and credentials already configured for the backend.
2. Send the candidate JSON unchanged as `candidate` and a description of the focus, lens and capture setup as `optical_setup` to `POST /api/v1/camera-profiles/{camera_id}/calibration/import`. Each distinct profile is retained in the registry. The latest imported version is active for that camera ID. Back up the registry before operational profile changes. Import only profiles that belong to the attached camera and optical setup.
3. In Monitor, enter the registered camera ID and its local USB index. Choose Automatic / registry to request the profile resolution. An explicit conflicting mode is rejected. The live telemetry identifies the active calibration as unvalidated until independent evidence is reviewed.
4. With the dashboard stream stopped, run the local capture probe from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path backend).Path
& 'C:\Users\Amigos\Projects\skyguard-fusion\backend\.venv\Scripts\python.exe' scripts/measure_usb_capture.py --index 1
```

The probe requests DirectShow/MJPG for each mode, checks decoded dimensions and records capture-only timing. Use the local index confirmed on that machine. A reported codec conflict is a capture error. Codecs without readback remain unconfirmed.

For angular validation, lock the camera and optical setup, measure independently surveyed target directions across the center and edges, and capture repeated observations in the exact profile mode. Compare horizontal and vertical estimates against the surveyed angles and keep signed bias, median absolute error and p95 by axis and location. Use separately timed camera-to-dashboard observations for end-to-end latency. Use a range reference for independent distance trials and verified object geolocation for Real Map trials.
## Phase 45 stereo verification

Run from the repository root with the dashboard camera stopped:

```powershell
$env:PYTHONPATH = (Resolve-Path backend).Path
$calibration = 'C:\Users\Amigos\Downloads\SkyGuard_Calibration_Install\SkyGuard_Camera_Calibration'
$python = 'C:\Users\Amigos\Projects\skyguard-fusion\backend\.venv\Scripts\python.exe'
$profile = Join-Path $calibration 'skyguard_stereo_v3_profile_provisional.json'
& $python scripts/verify_stereo_v3.py --profile $profile --capture --left 0 --right 1
& $python scripts/verify_stereo_v3.py --profile $profile --checkerboards (Join-Path $calibration 'stereo_v3_physical_validation\distance_1000mm') --reference-mm 1000 --max-epipolar-px 3
```

The capture probe verifies decoded dimensions and reported FOURCC without saving images. The timestamps measure host read return. A read pair remains invalid for moving-object triangulation until exposure timing and optical alignment have been independently characterized. Rectified XYZ is expressed in metres relative to the left optical center, with +X right, +Y down and +Z forward. The checkerboard replay instead measures range from the optical midpoint to the board center for comparison with the archived reference.

### Field validation procedure

- Fix the rig, mark the camera positions, record the active profile hash and capture settings, and check board residuals at the start and end. Use a timing target visible to both cameras to bound exposure-time offset under the actual frame rates and exposure settings.
- Use a controlled site with a designated pilot and a separate console operator. Establish an abort signal and agree on the test envelope before flight. Keep a static reference object with surveyed positions visible where practical.
- Record independent ground truth for a static target, hover, horizontal and vertical motion, approach and retreat, exit and re-entry, two targets if the site permits, and an out-of-class object. Keep per-scenario labels and timestamps. Mark interruptions and intervals with no ground truth.
- Record correct class and false detections, ID switches, tracking continuity, camera FPS, capture drops where observable, host and exposure skew as distinct metrics, dashboard latency, valid stereo fraction, distance error and empirical uncertainty coverage. Compare track ID and timestamp across both maps on matching frame events. Evaluate geographic agreement only with independently measured rig pose and target positions.
- Retain evidence and report each scenario independently. Static board acceptance and synthetic geometry tests do not establish moving-drone accuracy.
## Jury handoff

Use the Git branch `feature/skyguard-final-engineering`. Start the original operator console on localhost. The launcher configures the browser proxy and backend keys for the process. The UI does not require entering those keys.

From a fresh Windows checkout, install dependencies first:

```powershell
py -3.11 -m venv backend/.venv
$python = (Resolve-Path backend/.venv/Scripts/python.exe).Path
& $python -m pip install -r backend/requirements.txt -r backend/requirements-persistence.txt
npm.cmd --prefix frontend ci
```

Put the separately supplied `skyguard-drone-bird-t4-v1/best.pt` at `backend/models/skyguard-drone-bird-t4-v1/best.pt` or pass its local path with `-ModelPath`. The checkpoint used for the reference result has SHA-256 `eee9ea5271fa0e3505e3ff103d4e51202f234445e4c19c3482fa7aa00a122901`. The model and physical V3 profile are local inputs. Set `SKYGUARD_STEREO_PROFILE` to the supplied V3 JSON path when demonstrating stereo.

```powershell
$model = (Resolve-Path backend/models/skyguard-drone-bird-t4-v1/best.pt).Path
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/start_skyguard_operator.ps1 -Python $python -ModelPath $model
Start-Process 'http://127.0.0.1:5173/operator.html?view=monitor'
```

- Upload a jury-owned JPG, PNG or MP4 under Live observation. The reference image workflow and its recorded output are in [validation.json](backend/reports/multi-object-reference-provenance/validation.json). This single fixture verifies one DRONE and one BIRD with separate boxes. It does not establish benchmark accuracy on other media.
- For live capture, attach the two cameras and choose **Start stereo 0 + 1**. The verified local setup used Windows DirectShow, YUY2 and 640 × 480. Local camera indices can vary by machine. [Physical capture evidence](backend/reports/stereo-dual-visible-live.json)
- Open **Dual map views** for the rig reference form and both maps. [Shared-scene evidence](backend/reports/geospatial-live-shared-scene.json) records physical frame events. No validated real stereo XYZ, metric trajectory, object range, or WGS84 is available from those runs. The V3 profile is provisional. Confirm the timing and rig reference physically before using any projected position.
- To verify the software, run `python -m pytest -q` from `backend`, and `npm.cmd run build` plus `npx.cmd vitest run` from `frontend`.
