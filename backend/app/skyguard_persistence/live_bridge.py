"""SkyGuard existing main API + transactional persistence + exact-frame bridge.

Use with a single uvicorn worker, without --reload, on localhost.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
import math
import os
import secrets

import cv2
import numpy as np
from app.localization.rig_reference import RigReference
from app.phase43_45.maps import live_localization_scene
from app.localization.registered_frame_bearings import registered_optics
from app.localization.world_bearing import CameraOrientation, ray_to_world_direction
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text

from .database import Database
from .live_capture import LiveRecorder

# The original main.py imports the original camera router. We DO NOT replace
# YOLO, the tracker, security checks, or the camera session manager.
from app.api.live_camera import manager, start_live, stop_live, StartRequest

security = HTTPBearer(auto_error=False)


def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    expected = os.environ.get("SKYGUARD_DB_API_KEY", "")
    if not expected:
        raise HTTPException(503, "SKYGUARD_DB_API_KEY must be configured")
    if credentials is None:
        raise HTTPException(401, "Bearer token required")
    if not secrets.compare_digest(credentials.credentials.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(403, "Invalid bearer token")


def _frame_geometry(snapshot, jpeg, binding, detections) -> dict | None:
    """Project only the atomic JPEG using the calibration pinned to this session."""
    if not jpeg or binding is None or binding.camera_id != snapshot.camera_id or binding.session_id != snapshot.session_id:
        return None
    try:
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return None
        height, width = frame.shape
        record = binding.record()
        calibration = record.get("calibration") or {}
        optics = registered_optics(record, snapshot.camera_id, width, height)
        pose = record.get("orientation") or {}
        reference = pose.get("north_reference")
        orientation = None
        if (pose.get("source") == "calibration" and reference in ("true_north", "magnetic_north")
            and isinstance(pose.get("evidence_ref"), str) and pose["evidence_ref"].strip()):
            try:
                orientation = CameraOrientation(pose["yaw_deg"], reference, pose["pitch_deg"], pose["roll_deg"])
            except (KeyError, TypeError, ValueError):
                pass
        intrinsics = optics.intrinsics
        left = optics.pixel_to_direction(0, intrinsics.cy_px)
        right = optics.pixel_to_direction(width - 1, intrinsics.cy_px)
        top = optics.pixel_to_direction(intrinsics.cx_px, 0)
        bottom = optics.pixel_to_direction(intrinsics.cx_px, height - 1)
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            try:
                ray = optics.pixel_to_direction((x1 + x2) / 2, (y1 + y2) / 2)
            except ValueError:
                continue
            item["relative_bearing_deg"] = ray.yaw_offset_deg
            item["relative_elevation_deg"] = ray.elevation_offset_deg
            if orientation is not None:
                world = ray_to_world_direction(ray, orientation)
                item["azimuth_deg"] = world.azimuth_degrees
                item["north_reference"] = world.north_reference
        return {
            "horizontal_fov_deg": right.yaw_offset_deg - left.yaw_offset_deg,
            "vertical_fov_deg": top.elevation_offset_deg - bottom.elevation_offset_deg,
            "source": "session_pinned_registry_intrinsics",
            "actual_resolution": [width, height],
            "calibration_profile_id": calibration.get("profile_id"),
            "calibration_status": calibration.get("validation_status", "CALIBRATED_UNVALIDATED"),
            "physical_calibration_verified": False,
        }
    except (cv2.error, KeyError, TypeError, ValueError):
        return None


def _snapshot_data(recorder: LiveRecorder, snapshot, jpeg: bytes | None = None, binding=None, *, include_image=True) -> dict:
    state = snapshot.state
    frame_index = snapshot.frame_index
    detections = []
    for item in snapshot.detections:
        label = str(item.get("class", "")).upper()
        classification = label if label in ("DRONE", "BIRD") else "UNKNOWN"
        box = item.get("bbox", {})
        if not isinstance(box, dict):
            continue
        try:
            x, y, w, h = (float(box[v]) for v in ("x", "y", "width", "height"))
            if not all(map(math.isfinite, (x, y, w, h))) or min(x, y, w, h) < 0 or w <= 0 or h <= 0:
                continue
            bbox = [x, y, x + w, y + h]
        except (ValueError, KeyError, TypeError):
            continue
        detections.append({"classification": classification,
                           "confidence": item.get("confidence"),
                           "bbox": bbox, "track_id": None})
    # Use only exact detector-instance associations captured by live_capture;
    # bbox proximity alone never proves identity.
    event = recorder.observed_event(snapshot.session_id, frame_index)
    if event is not None:
        detections = []
        for obj in event.objects:
            record = {"classification": obj.classification,
                      "confidence": obj.confidence,
                      "bbox": list(obj.bbox), "track_id": obj.track_id if obj.classification != "UNKNOWN" else None}
            detections.append(record)
    geometry = _frame_geometry(snapshot, jpeg, binding, detections) if state == "running" else None
    tracks = [{**item, "frame_index": frame_index} for item in detections if item["track_id"] is not None]
    processing = snapshot.processing_seconds
    stereo = getattr(snapshot, "stereo", None) if state == "running" else None
    if stereo and not include_image:
        stereo = {key: value for key, value in stereo.items() if key != "right_image_base64"}
    captured_at = recorder.observation_time(snapshot.session_id, frame_index)
    response = {
        "state": state,
        "error": (snapshot.error if state == "error" and (snapshot.error or "").startswith("USB ") else None),
        "session_id": snapshot.session_id,
        "persistence_session_id": recorder.persisted_session_id(snapshot.session_id),
        "geometry": geometry,
        "stereo": stereo,
        "camera_id": snapshot.camera_id,
        "frame_index": frame_index,
        "captured_at": captured_at,
        "localization_scene": live_localization_scene(snapshot.session_id, frame_index, captured_at, stereo.get("objects", [])) if stereo else None,
        "timestamp_semantics": "host_publication_utc_not_sensor_timestamp",
        "fps": recorder.publication_fps(snapshot.session_id, frame_index),
        "fps_source": "host_frame_publication",
        "latency_ms": processing * 1000 if isinstance(processing, (float, int)) and processing >= 0 else None,
        "detections": detections,
        "tracks": tracks,
        "source": "SKYGUARD LIVE CAMERA - synchronized server frame",
        "image_mime": "image/jpeg" if include_image and jpeg is not None and state == "running" else None,
        "image_base64": base64.b64encode(jpeg).decode("ascii") if include_image and jpeg is not None and state == "running" else None,
        "recorder": recorder.health(),
    }
    return response


def mount_live_bridge(app: FastAPI, database: Database) -> None:
    recorder = LiveRecorder(database)
    api = APIRouter(prefix="/api/persistence/live", tags=["SkyGuard live persistence bridge"], dependencies=[Depends(authorize)])

    reference_path = Path(os.environ.get("SKYGUARD_RIG_REFERENCE", ".skyguard_data/stereo-rig-reference.json"))

    def read_reference():
        try:
            raw = reference_path.read_bytes()
        except FileNotFoundError:
            return {"status": "GEO_REFERENCE_MISSING", "reference": None, "id": None}
        try:
            reference = RigReference.model_validate_json(raw)
        except ValueError as exc:
            raise HTTPException(500, "Invalid local rig reference") from exc
        return {"status": reference.status(), "reference": reference.model_dump(mode="json"),
                "id": hashlib.sha256(raw).hexdigest()[:16]}

    @api.get("/rig-reference")
    def rig_reference():
        return read_reference()

    @api.post("/rig-reference")
    def save_rig_reference(reference: RigReference):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=reference_path.parent, delete=False) as temporary:
            temporary.write(reference.model_dump_json().encode())
        temporary_path = Path(temporary.name)
        try:
            temporary_path.replace(reference_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return read_reference()

    @api.post("/start")
    def start_original_camera(payload: StartRequest):
        # Exact existing source validation, model availability and session guards.
        return start_live(payload)

    @api.post("/stop")
    def stop_original_camera():
        return stop_live()

    @api.get("/frame")
    def synchronized_live_frame(include_image: bool = True):
        snapshot, jpeg, binding = manager.latest_bound_frame()
        return _snapshot_data(recorder, snapshot, jpeg, binding, include_image=include_image)

    @api.get("/status")
    def synchronized_live_status():
        snapshot = manager.status()
        return _snapshot_data(recorder, snapshot)

    @api.get("/health")
    def live_bridge_health():
        status = recorder.health()
        status["live_state"] = manager.status().state
        return status

    app.include_router(api)

    # Avoid mutating existing lifespan/routers. No event is generated by frontend polling.
    # Uvicorn startup hooks install the non-blocking capture callback before live start.
    @app.on_event("startup")
    def start_bridge():
        with database.engine.connect() as connection:
            # Never automatically create/change production schema; Alembic required.
            connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).scalar_one()
        recorder.start(manager)

    @app.on_event("shutdown")
    def stop_bridge():
        try:
            manager.stop()
            recorder.stop()
        finally:
            database.dispose()
