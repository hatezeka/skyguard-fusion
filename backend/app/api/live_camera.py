"""Local-only SkyGuard live camera API."""

import base64
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StrictInt, StrictStr

from app.api.camera_management import registry
from app.camera_management.core import CameraError
from app.detection.live_session import LiveSessionManager
from app.detection.yolo import YOLODetectionEngine
from app.detection.motion import MotionDetectionEngine
from app.detection.airborne_fusion import AirborneFusionDetectionEngine
from app.detection.advanced_tracking_adapter import AdvancedTrackingAdapter
from app.video.source import VideoSourceConfig


router = APIRouter(prefix="/api/v1/live", tags=["Live Camera"])

manager = LiveSessionManager()
start_lock = threading.Lock()

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "skyguard-drone-bird-t4-v1"
    / "best.pt"
)


class StartRequest(BaseModel):
    camera_index: int = Field(default=0, ge=0, le=4)
    confidence: float = Field(default=0.35, ge=0.01, le=1.0)
    fusion_enabled: bool = False
    source_type: str = "usb"
    source_location: StrictInt | StrictStr | None = None
    camera_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )


def validate_camera_identity(request: StartRequest) -> str | None:
    if request.camera_id is None:
        return None

    try:
        profile = registry().get(request.camera_id)
    except CameraError as exc:
        raise HTTPException(
            status_code=404,
            detail="Registered camera not found",
        ) from exc

    if profile["source_type"] != request.source_type:
        raise HTTPException(
            status_code=422,
            detail="Camera source type does not match registry",
        )

    registered_location = profile.get("source_location")

    if registered_location is not None:
        if request.source_type == "usb":
            requested_location = (
                request.camera_index
                if request.source_location is None
                else request.source_location
            )
        else:
            requested_location = request.source_location

        if requested_location != registered_location:
            raise HTTPException(
                status_code=422,
                detail="Camera source does not match registry",
            )

    return request.camera_id


def resolve_live_source(request: StartRequest) -> VideoSourceConfig:
    if request.source_type == "usb":
        location = (
            request.camera_index
            if request.source_location is None
            else request.source_location
        )
        if type(location) is not int or not 0 <= location <= 4:
            raise HTTPException(422, "USB camera index must be 0-4")
        return VideoSourceConfig("usb", location)

    if request.source_type == "rtsp":
        if os.environ.get("SKYGUARD_ENABLE_RTSP_LIVE") != "1":
            raise HTTPException(403, "RTSP live input is disabled")

        location = request.source_location
        if not isinstance(location, str):
            raise HTTPException(422, "RTSP URL required")

        parsed = urlsplit(location)
        if (
            parsed.scheme.lower() != "rtsp"
            or not parsed.hostname
            or parsed.fragment
            or any(ch.isspace() for ch in location)
        ):
            raise HTTPException(422, "Invalid RTSP URL")

        return VideoSourceConfig("rtsp", location)

    raise HTTPException(422, "Unsupported source type")


@router.post("/start")
def start_live(request: StartRequest):
    if not start_lock.acquire(blocking=False):
        raise HTTPException(409, "Camera start already in progress")

    try:
        if manager._thread is not None and manager._thread.is_alive():
            raise HTTPException(409, "Camera session already active")

        validate_camera_identity(request)
        # Captured once before the manager creates a fresh session_id.
        # The manager canonicalizes and freezes only optical/pose fields.
        session_calibration_record = (
            registry().get(request.camera_id) if request.camera_id is not None else None
        )

        configured_path = os.environ.get("SKYGUARD_MODEL_PATH")
        model_path = Path(configured_path) if configured_path else MODEL_PATH
        if not model_path.is_file():
            raise HTTPException(503, "YOLO model unavailable")

        detector = YOLODetectionEngine(
            model_path=str(model_path),
            confidence_threshold=request.confidence,
        )

        config = resolve_live_source(request)

        if request.fusion_enabled:
            detector = AirborneFusionDetectionEngine(
                specialized_detector=detector,
                candidate_detector=MotionDetectionEngine(),
            )
            tracker = AdvancedTrackingAdapter()
            manager.start(
                config,
                detector,
                camera_id=request.camera_id,
                calibration_record=session_calibration_record,
                tracker=tracker,
            )
        else:
            manager.start(
                config,
                detector,
                camera_id=request.camera_id,
                calibration_record=session_calibration_record,
            )

        return asdict(manager.status())
    finally:
        start_lock.release()


@router.post("/stop")
def stop_live():
    return asdict(manager.stop())


@router.get("/status")
def live_status():
    return asdict(manager.status())


@router.get("/detections")
def live_detections():
    snapshot = manager.status()

    return {
        "state": snapshot.state,
        "frame_index": snapshot.frame_index,
        "camera_id": snapshot.camera_id,
        "session_id": snapshot.session_id,
        "detections": snapshot.detections,
        "tracks": snapshot.tracks,
        "ambiguous_tracks": snapshot.ambiguous_tracks,
        "trajectories": snapshot.trajectories,
        "processing_seconds": snapshot.processing_seconds,
        "error": snapshot.error,
    }



@router.get("/frame")
def live_frame():
    snapshot, jpeg = manager.latest_frame()

    if snapshot.state != "running" or jpeg is None:
        raise HTTPException(
            status_code=409,
            detail="Synchronized frame is unavailable",
        )

    return {
        "state": snapshot.state,
        "frame_index": snapshot.frame_index,
        "session_id": snapshot.session_id,
        "camera_id": snapshot.camera_id,
        "image_mime": "image/jpeg",
        "image_base64": base64.b64encode(jpeg).decode("ascii"),
        "tracks": snapshot.tracks,
        "trajectories": snapshot.trajectories,
    }


from app.localization.camera_profile import CameraGeoProfile
from app.localization.live_frame_bearings import (
    LiveBearingRequest, derive_live_frame_bearings,
)


@router.post("/frame/bearings")
def live_frame_bearings(payload: LiveBearingRequest):
    """Compute directions for an exact, atomically retrieved live frame."""
    snapshot, jpeg = manager.latest_frame()
    if snapshot is None or getattr(snapshot, "state", None) != "running" or jpeg is None:
        raise HTTPException(409, "Synchronized frame is unavailable")
    if (
        getattr(snapshot, "camera_id", None) != payload.camera_id
        or getattr(snapshot, "frame_index", None) != payload.expected_frame_index
    ):
        raise HTTPException(409, "Active camera or frame does not match request")

    if getattr(snapshot, "session_id", None) != payload.expected_session_id:
        raise HTTPException(409, "Active session does not match request")

    # This is the existing /api/v1/geo camera profile, not a made-up position.
    from app.api import geo
    record = geo.get_camera_profile(payload.camera_id)
    try:
        profile = CameraGeoProfile(
            camera_id=record["camera_id"],
            latitude=record["latitude"],
            longitude=record["longitude"],
            heading_degrees=record["heading_degrees"],
            horizontal_fov_degrees=record["horizontal_fov_degrees"],
            coverage_radius_m=record["coverage_radius_m"],
        )
        return derive_live_frame_bearings(
            snapshot, jpeg, request=payload, profile=profile
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


from app.localization.registered_frame_bearings import (
    RegisteredBearingRequest, derive_registered_frame_bearings,
)


@router.post("/frame/bearings/registered/pinned")
def live_registered_frame_bearings_pinned(payload: RegisteredBearingRequest):
    """Use ONLY the immutable optical snapshot selected when session started.

    Unlike /registered, this route never reloads the mutable camera registry.
    Profile-to-frame provenance is software-verified; physical survey is not.
    """
    snapshot, jpeg, binding = manager.latest_bound_frame()
    if snapshot is None or getattr(snapshot, "state", None) != "running" or jpeg is None:
        raise HTTPException(409, "Synchronized frame is unavailable")
    if (
        getattr(snapshot, "camera_id", None) != payload.camera_id
        or getattr(snapshot, "frame_index", None) != payload.expected_frame_index
        or getattr(snapshot, "session_id", None) != payload.expected_session_id
    ):
        raise HTTPException(409, "Active camera, frame, or session does not match request")
    if (
        binding is None
        or binding.camera_id != payload.camera_id
        or binding.session_id != payload.expected_session_id
    ):
        raise HTTPException(409, "No calibration snapshot is bound to this session")
    try:
        data = derive_registered_frame_bearings(
            snapshot, jpeg, request=payload, registry_record=binding.record()
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    data.update({
        "profile_source": "camera_registry_session_snapshot",
        "calibration_snapshot_sha256": binding.sha256,
        "profile_bound_to_session_verified": True,
        "profile_bound_to_frame_verified": True,
        "physical_calibration_verified": False,
    })
    return data


@router.post("/frame/bearings/registered")
def live_registered_frame_bearings(payload: RegisteredBearingRequest):
    """Directions using current registered intrinsics and orientation.

    Registry provenance is not physical proof or session-pinned calibration.
    """
    snapshot, jpeg = manager.latest_frame()
    if snapshot is None or getattr(snapshot, "state", None) != "running" or jpeg is None:
        raise HTTPException(409, "Synchronized frame is unavailable")
    if (
        getattr(snapshot, "camera_id", None) != payload.camera_id
        or getattr(snapshot, "frame_index", None) != payload.expected_frame_index
        or getattr(snapshot, "session_id", None) != payload.expected_session_id
    ):
        raise HTTPException(409, "Active camera, frame, or session does not match request")

    from app.api import camera_management
    from app.camera_management.core import CameraError
    try:
        record = camera_management.registry().get(payload.camera_id)
    except CameraError as exc:
        code = 404 if str(exc) == "unknown camera" else 409
        raise HTTPException(code, detail=str(exc)) from exc

    try:
        return derive_registered_frame_bearings(
            snapshot, jpeg, request=payload, registry_record=record
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def mjpeg_frames():
    last_frame = None

    while True:
        snapshot, jpeg = manager.latest_frame()

        if snapshot.state in ("idle", "stopped", "error"):
            break

        if jpeg is not None and snapshot.frame_index != last_frame:
            last_frame = snapshot.frame_index

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

        time.sleep(0.1)


@router.get("/stream")
def live_stream():
    snapshot = manager.status()

    if snapshot.state not in ("starting", "running"):
        raise HTTPException(409, "Camera session is not active")

    return StreamingResponse(
        mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )

# Phase 40.1 â€” the EXISTING live router inherits the existing app auth policy.
from app.localization.live_map_scene import LiveMapSceneRequest, build_live_map_scene
from app.camera_management.core import CameraError


@router.post("/map/scene")
def live_map_scene(payload: LiveMapSceneRequest):
    """A session-pinned direction scene with CURRENT, explicitly unbound camera GPS.

    Lines are cartographic bearing GUIDES only, never target positions/range.
    The mutable geolocation record is reported as NOT session-bound.
    """
    snapshot, jpeg, binding = manager.latest_bound_frame()
    if snapshot is None or getattr(snapshot, "state", None) != "running" or jpeg is None:
        raise HTTPException(409, "Synchronized frame is unavailable")
    if (
        getattr(snapshot, "camera_id", None) != payload.camera_id
        or getattr(snapshot, "frame_index", None) != payload.expected_frame_index
        or getattr(snapshot, "session_id", None) != payload.expected_session_id
    ):
        raise HTTPException(409, "Active camera, frame, or session does not match request")
    if (
        binding is None
        or binding.camera_id != payload.camera_id
        or binding.session_id != payload.expected_session_id
    ):
        raise HTTPException(409, "No calibration snapshot is bound to this session")
    try:
        # All directions use the same immutable optics/pose/JPEG snapshot.
        bearings = derive_registered_frame_bearings(
            snapshot,
            jpeg,
            request=RegisteredBearingRequest(
                camera_id=payload.camera_id,
                expected_frame_index=payload.expected_frame_index,
                expected_session_id=payload.expected_session_id,
                north_reference=payload.north_reference,
            ),
            registry_record=binding.record(),
        )
        # Only camera location is read from CURRENT registry. This is NOT a
        # location-at-capture proof and the scene explicitly states so.
        location_profile = registry().get(payload.camera_id)
        from app.localization.live_scene_dual_map_adapter_v1 import attach_dual_map_state
        return attach_dual_map_state(build_live_map_scene(
            bearings,
            location_profile,
            calibration_sha256=binding.sha256,
            north_reference=payload.north_reference,
            expected_camera_id=payload.camera_id,
            expected_session_id=payload.expected_session_id,
            expected_frame_index=payload.expected_frame_index,
        ))
    except CameraError as exc:
        raise HTTPException(409, "Camera location profile is unavailable") from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(409, detail=str(exc)) from exc

# Phase 40.2: atomic latest scene for live UI; existing protected router auth applies.
from typing import Literal as _MapNorthLiteral
from fastapi import Query as _MapQuery


@router.get("/map/scene/latest")
def live_map_scene_latest(
    north_reference: _MapNorthLiteral["true_north", "magnetic_north"] = _MapQuery(...)
):
    """Return one internally synchronized current frame, never a guessed frame.

    Still does NOT claim target GPS/range or a surveyed camera location.
    """
    snapshot, jpeg, binding = manager.latest_bound_frame()
    if (
        snapshot is None
        or getattr(snapshot, "state", None) != "running"
        or jpeg is None
        or not getattr(snapshot, "camera_id", None)
        or not getattr(snapshot, "session_id", None)
        or type(getattr(snapshot, "frame_index", None)) is not int
    ):
        raise HTTPException(409, "Synchronized registered camera frame is unavailable")
    if (
        binding is None
        or binding.camera_id != snapshot.camera_id
        or binding.session_id != snapshot.session_id
    ):
        raise HTTPException(409, "No calibration snapshot is bound to this session")
    try:
        bearings = derive_registered_frame_bearings(
            snapshot,
            jpeg,
            request=RegisteredBearingRequest(
                camera_id=snapshot.camera_id,
                expected_frame_index=snapshot.frame_index,
                expected_session_id=snapshot.session_id,
                north_reference=north_reference,
            ),
            registry_record=binding.record(),
        )
        location_profile = registry().get(snapshot.camera_id)
        return build_live_map_scene(
            bearings,
            location_profile,
            calibration_sha256=binding.sha256,
            north_reference=north_reference,
            expected_camera_id=snapshot.camera_id,
            expected_session_id=snapshot.session_id,
            expected_frame_index=snapshot.frame_index,
        )
    except CameraError as exc:
        raise HTTPException(409, "Camera location profile is unavailable") from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(409, detail=str(exc)) from exc


# Phase 41.2: on-demand, evidence-limited alert evaluation.
# Registered on the EXISTING protected live router; no public bypass route.
from app.alerts.live_advisory import LiveAdvisoryBridge as _LiveAdvisoryBridge
_live_advisory_bridge = _LiveAdvisoryBridge()


@router.post("/alerts/evaluate/latest")
def live_alert_evaluate_latest():
    """Evaluate only the server-selected latest snapshot, never client evidence.

    Only model/tracker advisories; no drone GPS, range, perimeter breach,
    autonomous action or durable archive. Polling can skip camera frames.
    """
    try:
        return _live_advisory_bridge.evaluate(manager.status)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(409, "Live advisory snapshot unavailable or stale") from exc
