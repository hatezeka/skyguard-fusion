# SkyGuard FUSION — Phase 42.1: offline multi-object benchmark contract

This is a **real evaluation algorithm** for supplied predictions and annotated ground truth. It **does not** connect to the camera, run YOLO, download datasets, or assert empirical detector performance. The included three-frame crossing fixture is **synthetic** and must never be reported as dataset accuracy.

## Purpose

With labeled real footage and corresponding model/tracker predictions, measure:

- Per-class detection/classification TP, FP, FN, precision, recall, F1;
- A class-agnostic, globally one-to-one bounding-box IoU association and confusion matrix (misclassifications remain visible);
- Multi-drone concurrency observed in the supplied input;
- Predicted track-ID switches against annotated ground-truth drone identity, per sequence, within a configurable frame gap.

**Not** IDF1, HOTA, MOTA, real-world target identity confirmation, or range/GPS. Measure those separately using suitable protocols and sensors. A reported "max simultaneous drones" means **only maximum in the provided benchmark clip**, not a guaranteed system capacity.

## Input: one strict UTF-8 JSON object per frame

```json
{"sequence_id":"clip_001","frame_index":0,"width":640,"height":480,"ground_truth":[{"id":"gt_drone_1","class":"drone","bbox_xyxy":[10,20,40,50]}],"predictions":[{"track_id":"tracker_1","class":"drone","bbox_xyxy":[11,20,41,50],"confidence":0.94}]}
```

- The `per_sequence` result keys use `s:` for string identities and `i:` for integer identities to avoid collisions.
- `sequence_id` and GT `id` are annotation identities, **not GPS coordinates**. Track IDs are predictions from the tracker; no IDs are inferred.
- Boxes are pixel-space `[x1,y1,x2,y2]`, strictly positive area and within `width`/`height`.
- Zero-based `frame_index` must increase within each sequence. Multiple sequences may be interleaved.
- GT annotations must be independent of model predictions. Only frames included in JSONL are evaluated. If the model omits detections, include those frames with `predictions: []`.
- Strict schema rejects unknown fields, including fabricated `range_m`, `gps`, or geographic breach claims.
- Drop predictions below `confidence_threshold` (default 0.25); match bbox IoU at default 0.50. A classification error yields both FP for predicted class and FN for actual class. A GT/pred pair is one-to-one globally optimal for maximum eligible matches, then IoU.
- Frame count is the number of **submitted annotated records**; evaluator does not check actual video coverage. It is the user's job to include complete annotated sequences.
- Run representative video from the Halmstad, DUT Anti-UAV, UAVSwarm, ETH Multi-view, or other appropriately licensed datasets; independently verify licenses and annotation mapping. **Nothing is downloaded automatically.**

## Running on Windows PowerShell

From repository root, using the project's Python virtual environment:

```powershell
$env:PYTHONPATH = (Resolve-Path 'backend').Path
& 'backend\.venv\Scripts\python.exe' -m app.evaluation.mot_benchmark --input backend/examples/phase42_1_synthetic_crossing.jsonl
```

Use a unique `--output C:\...\results.json` for reproducible machine-readable results. No file overwrite is allowed. Change `--confidence-threshold`, `--iou-threshold` or `--max-switch-gap` explicitly to compare scenarios. The provided example must be labeled **synthetic test fixture**, not a field benchmark.

## Evidence requirements before interpreting results as challenge performance

1. Verify dataset license, source, split, annotation completeness and actual video/model matching.
2. Record model checkpoint hash, inference version, camera/video resolution, operating hardware, latency and frame throughput **separately** from this evaluator.
3. Evaluate unseen sequences and crossing/occlusion cases. Report per-video results and error cases, false positives for birds and objects, and identity switches.
4. For geographic target position/distance, use an independently validated ranging method or multi-view calibration. A single monocular bearing does not provide reliable distance or a drone GPS fix.
