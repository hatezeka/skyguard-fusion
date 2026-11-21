# Phase 42.4 — bounded local video/camera throughput and latency

**Status:** an instrumented profiler and test suite. Passing this phase's unit tests does **not** certify a physical camera, actual drone accuracy, GPU performance, lost frames, or the SkyGuard production tracker.

This phase deliberately does not edit the live camera session or start cameras by default. It has three explicit modes:

1. **Bundled artificial timing trace** (no camera, video, model or network). For a reproducible *mechanism check only*:

   ```powershell
   cd C:\Users\Amigos\Projects\skyguard-fusion
   & backend\.venv\Scripts\python.exe -m app.evaluation.camera_performance --trace backend\examples\phase42_4_SYNTHETIC_timing.jsonl --synthetic-self-test --output "$env:TEMP\skyguard-42-4-synthetic-unique.json"
   ```

2. **Local licensed video, decoding only** (requires local OpenCV):

   ```powershell
   & backend\.venv\Scripts\python.exe -m app.evaluation.camera_performance --video "C:\path\to\owned-video.mp4" --confirm-license --max-frames 300 --max-seconds 30 --target-fps 15 --output "$env:TEMP\skyguard-42-4-decode-unique.json" --trace-output "$env:TEMP\skyguard-42-4-decode-unique.jsonl"
   ```

3. **Physical camera, only after operator opt-in**:

   ```powershell
   & backend\.venv\Scripts\python.exe -m app.evaluation.camera_performance --camera-index 0 --confirm-camera --max-frames 300 --max-seconds 30 --target-fps 15 --output "$env:TEMP\skyguard-42-4-camera-unique.json"
   ```

**Optional trusted checkpoint:** append `--weights "C:\path\to\trusted\best.pt" --confirm-trusted-checkpoint --device cpu` (or a CUDA device index such as `0` when supported). It runs **Ultralytics YOLO.predict as an independent reference**, not the actual SkyGuard live tracker. Do not claim that its FPS is the production pipeline FPS. The reference must never be auto-downloaded. No install or model training happens in the acceptance script. Only run `.pt` files from trusted sources: PyTorch checkpoint loading can execute code.

## Measurements, not inferences

- `wall_throughput_fps` = frames actually read/processed divided by elapsed wall time of bounded run. Includes capture read and, when requested, model prediction.
- `interarrival_fps` uses differences between the timestamp when each measured frame finished. This is **not** sensor-native frame rate.
- `read_latency_ms`, `inference_latency_ms` (optional), `processing_cycle_ms` report mean and nearest-rank p50/p95/p99/max.
- `long_interarrival_gap_count_not_proven_drops` detects unusually long completion gaps; **not** actual dropped frames. `actual_dropped_frames` is explicitly `null` until independent hardware frame numbering is available.
- `target_fps_budget` and `over_budget_frame_count` compare measured per-frame cycles with an operator-chosen throughput budget.
- `fps_reported_by_capture_metadata_not_measured` is driver/video metadata, not independently validated sensor FPS.
- Raw images/video are **never** saved by this profiler. Optional `--trace-output` saves *only timing observations* as a new JSONL file. Existing output paths are never overwritten. Check provenance and source hashes before comparing runs.
- For a video file, no independent ground-truth annotations are required because **this does not measure detection precision/recall or ID stability**; use 42.1–42.3 with independent GT for those measurements.
- Neither a monocular ray nor timing data yields a target GPS position, physical range, or verified perimeter violation.

## Operational and security caveats

Camera access is opt-in and may prompt operating-system privacy permissions. `--max-frames` and `--max-seconds` bound the loop, but a blocked hardware `VideoCapture.read()` call may ignore the time budget: a hanging vendor driver is outside this profiler's control. Use a supported, trusted camera in a supervised test. Errors abort without publishing a successful report; partial data are not accepted as a completed benchmark. Capturing video in RAM still exposes potentially sensitive surroundings to the local process.

Physical latency/FPS measurements must be repeated with documented camera make/model, lens, resolution, GPU/CPU, ambient lighting, distance to objects, session duration, and model SHA256. Performance observed with Ultralytics.predict is not proof that SkyGuard's own live API, model+tracker, browser UI, or alerting achieves that FPS.

The acceptance script adds only `camera_performance.py`, tests, this document and a named **SYNTHETIC** fixture. It tests the parser, fake camera/model, and earlier modules without accessing real camera hardware. It does not mutate Git staging, change dependencies, install anything, or perform Git commits.
