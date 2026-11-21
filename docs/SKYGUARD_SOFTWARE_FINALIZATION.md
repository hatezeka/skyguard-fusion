# SkyGuard FUSION â€” Software finalization and scientific evidence

This release kit is not a new YOLO model and **does not** train on non-existent GT or claim field acceptance. It upgrades the production tracking path to reject UNKNOWN for tracking and avoid cross-class identity reuse, while retaining UNKNOWN detections for the database.

## Reproducible evidence pipeline

1. Ensure local `feature/skyguard-mvp` branch. Do not commit/push without authorization.
2. Run `SkyGuard_Software_Finalization_OneCommand.ps1` on Windows PowerShell. It creates backups outside Git, applies exact-source guarded patches, and creates new additive science/evidence modules.
3. Run complete pytest, complete Vitest, Oxlint and Vite production build.
4. Run SQLite integrity and scientific gate report.
5. Optional three-class dataset acceptance is available with `-DataYaml` and `-SourceManifest`. Review licenses and bounding boxes; with `-TrainThreeClass` and a verified base model, opt in to a training run (can be long). Without those files model acceptance stays BLOCKED.
6. For **real recorded media** only: give `-VideoPath`, `-ConfirmMediaRights`, optional `-GroundTruthJson`, `-ModelPath`. This runs actual production YOLO + tracker + recorder into a **temporary** DB and tests persistence after reopening.
7. Save resulting acceptance JSON/Markdown from the reported directory; report BLOCKED criteria to the jury clearly.

## Source-independent validation requirements

- An independently reviewed and licensed test dataset with source-disjoint splits.
- Three-class DRONE/BIRD/OTHER airborne model for robust UNKNOWN handling, with held-out novel classes. Two-class Colab V1 alone cannot establish robust unknown rejection.
- Per-class confusion, false drone alarms over bird/plane/helicopter, P/R/mAP stratified by apparent size/distance, and evidence of unseen video.
- IDF1/HOTA/ID switches/fragmentation with track-ID-labelled ground truth in crossing/occlusion multi-object video. Scoring helper provides IDF1 only when IDs are available. It deliberately does not invent HOTA.
- Paired map (WGS84/ENU) correctness tied to same frame/session/ID, licensed basemap attribution, no GPS inferred from a monocular bearing.
- True FPS/latency P50/95/99, CPU/GPU/RAM, dropped-frame measurement and low-cost dated BOM on final hardware.
- Black-box abuse tests, RTSP SSRF/rate bounds, secrets management, crash/restart/recovery, DB retention and encrypted production authentication.
- Controlled physical USB camera/drone trials are Phase 45, **not** part of this kit.

## Ground truth JSON format (optional)

```json
[
  {"sequence_id":"clip","frame_index":0,"objects":[{"class":"BIRD","bbox":[10,20,40,50],"track_id":"human-verified-bird-1"}]},
  {"sequence_id":"clip","frame_index":1,"objects":[]}
]
```

Above values are **examples**, not actual ground truth. Include **every frame** in the clip, including empty frames. Track-ID GT must be reviewed by humans. Use the original decoded video frame coordinate system for boxes, accounting for any re-encoding transforms.

## Persistent and temporary data

Operational DB: `backend/.skyguard_data/skyguard.sqlite3` remains unchanged by video acceptance. Scratch video and scratch SQLite are automatically destroyed after run. Evidence reports are created outside the repository (in Downloads). Never stage `.skyguard_data`, `.env`, backup files, secrets or unlicensed video in Git.

## Authentication notice

A local development bearer key previously appeared in conversation logs. Rotate it; do not reuse or embed it in Vite variables or bundles. Do not confuse a local single-user key with a production identity system.

## Three-class source provenance manifest example

```json
{
  "items": [
    {"split":"train", "image":"train/images/example.jpg", "label":"train/labels/example.txt", "source_group":"source_video_id", "license":"verified-license-name", "annotation_reviewed":true}
  ]
}
```

Illustrative structure only, **not** a verified dataset. Every train/val/test image must be represented, grouped by source video/site/object. The module rejects source leakage, duplicate files across splits, absent human review, absent third-class bounding boxes, invalid YOLO annotations and unverified rights. Training performance remains unproven until independent video evaluation.
