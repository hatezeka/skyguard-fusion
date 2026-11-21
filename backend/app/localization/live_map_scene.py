"""Phase 40.1: safe live-map scene from a session-pinned frame bearing result.

Only the camera's DECLARED location can be shown on a world map. The camera
location is read from the MUTABLE registry and is NOT session-bound yet.
No target range/GPS can be observed from a single camera. Bearing guides are
200 m DISPLAY-ONLY segments, and are not target positions or travel paths.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Literal

from pydantic import BaseModel, Field, StrictInt, StrictStr


class LiveMapSceneRequest(BaseModel):
    camera_id: StrictStr = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    expected_frame_index: StrictInt = Field(ge=0)
    expected_session_id: StrictStr = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    north_reference: Literal["true_north", "magnetic_north"]


_ALLOWED_LOCATION_SOURCES = frozenset({"unavailable", "device", "external_gnss", "manual", "simulation"})
_ALLOWED_CLASSES = frozenset({"drone", "bird", "unknown"})
_DISPLAY_GUIDE_LENGTH_M = 200.0
_EARTH_MEAN_RADIUS_M = 6_371_008.8


def _number(value: object, field: str, low: float, high: float) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {field}")
    return float(value)


def _camera_location(camera_id: str, profile: Mapping) -> dict | None:
    if not isinstance(profile, Mapping) or profile.get("camera_id") != camera_id:
        raise ValueError("Camera registry identity mismatch")
    geo = profile.get("geolocation")
    if not isinstance(geo, Mapping):
        raise ValueError("Camera geolocation record is unavailable")
    source = geo.get("source")
    if source not in _ALLOWED_LOCATION_SOURCES:
        raise ValueError("Unsupported camera location source")
    lat, lon = geo.get("latitude"), geo.get("longitude")
    if lat is None and lon is None:
        return None
    if lat is None or lon is None or source == "unavailable":
        raise ValueError("Incomplete camera location metadata")
    lat = _number(lat, "camera latitude", -90, 90)
    lon = _number(lon, "camera longitude", -180, 180)
    accuracy = geo.get("horizontal_accuracy_m")
    if accuracy is not None:
        accuracy = _number(accuracy, "camera location accuracy", 0, 100_000_000)
    # Do not infer a timestamp; the registry can contain manual/simulated values.
    captured_at = geo.get("captured_at")
    if captured_at is not None and not isinstance(captured_at, str):
        raise ValueError("Invalid camera location timestamp")
    return {
        "latitude": lat, "longitude": lon,
        "source": source, "horizontal_accuracy_m": accuracy,
        "captured_at": captured_at,
        "survey_verified": False,
        "bound_to_session_verified": False,
    }


def _display_endpoint(lat: float, lon: float, azimuth: float) -> tuple[float, float] | None:
    """Fixed display guide, never a measured range or a target position.

    Avoid a visually misleading polyline at polar regions or across dateline.
    """
    if abs(lat) >= 89.99:
        return None
    bearing = math.radians(azimuth)
    north = math.radians(lat)
    east = math.radians(lon)
    delta = _DISPLAY_GUIDE_LENGTH_M / _EARTH_MEAN_RADIUS_M
    sin_lat2 = math.sin(north) * math.cos(delta) + math.cos(north) * math.sin(delta) * math.cos(bearing)
    lat2 = math.asin(max(-1.0, min(1.0, sin_lat2)))
    lon2 = east + math.atan2(
        math.sin(bearing) * math.sin(delta) * math.cos(north),
        math.cos(delta) - math.sin(north) * math.sin(lat2),
    )
    normalized_lon = ((math.degrees(lon2) + 180) % 360) - 180
    # Crossing +/-180Â° meridian would create a globe-spanning straight segment.
    if abs(normalized_lon - lon) > 180:
        return None
    return math.degrees(lat2), normalized_lon


def build_live_map_scene(
    pinned_bearings: Mapping,
    location_profile: Mapping,
    *,
    calibration_sha256: str,
    north_reference: Literal["true_north", "magnetic_north"],
    expected_camera_id: str,
    expected_session_id: str,
    expected_frame_index: int,
) -> dict:
    """Create GeoJSON camera/guide features and non-geolocated observations.

    Required upstream provenance: one running frame, one session-pinned optical
    calibration. The geographic camera location still comes from the CURRENT
    registry and must be displayed with that limitation, never as surveyed.
    """
    if not isinstance(pinned_bearings, Mapping):
        raise ValueError("Pinned bearing result is missing")
    camera_id = pinned_bearings.get("camera_id")
    session_id = pinned_bearings.get("session_id")
    frame_index = pinned_bearings.get("frame_index")
    if not isinstance(camera_id, str) or not camera_id or len(camera_id) > 128:
        raise ValueError("Invalid scene camera identity")
    if not isinstance(session_id, str) or len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
        raise ValueError("Invalid scene session identity")
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("Invalid scene frame index")
    if (camera_id != expected_camera_id or session_id != expected_session_id or
            frame_index != expected_frame_index):
        raise ValueError("Map scene identity does not match requested frame")
    if not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64 or any(c not in "0123456789abcdef" for c in calibration_sha256):
        raise ValueError("Invalid calibration fingerprint")
    if north_reference not in ("true_north", "magnetic_north"):
        raise ValueError("Explicit north reference required")
    if pinned_bearings.get("north_reference") != north_reference:
        raise ValueError("Mismatched north reference")

    directions = pinned_bearings.get("directions")
    if not isinstance(directions, list) or len(directions) > 512:
        raise ValueError("Invalid detection-bearing array")
    location = _camera_location(camera_id, location_profile)
    features = []
    if location:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [location["longitude"], location["latitude"]]},
            "properties": {
                "kind": "camera", "camera_id": camera_id,
                "location_source": location["source"],
                "survey_verified": False,
                "location_bound_to_session_verified": False,
            },
        })

    observations = []
    used_indices: set[int] = set()
    for item in directions:
        if not isinstance(item, Mapping):
            raise ValueError("Invalid detection-bearing item")
        index = item.get("detection_index")
        if type(index) is not int or index < 0 or index in used_indices:
            raise ValueError("Invalid or duplicate detection index")
        used_indices.add(index)
        object_class = item.get("object_class")
        if object_class not in _ALLOWED_CLASSES:
            raise ValueError("Unsupported detection class")
        confidence = _number(item.get("confidence"), "detection confidence", 0, 1)
        direction = item.get("direction")
        if not isinstance(direction, Mapping) or direction.get("north_reference") != north_reference:
            raise ValueError("Invalid or mismatched world direction")
        azimuth = direction.get("azimuth_degrees")
        if azimuth is not None:
            azimuth = _number(azimuth, "azimuth", 0, 360)
            if azimuth >= 360:
                raise ValueError("Azimuth must be below 360 degrees")
        elevation = _number(direction.get("elevation_degrees"), "elevation", -90, 90)
        observations.append({
            "detection_index": index,
            "object_class": object_class,
            "classification_is_confirmed": False,
            "confidence": confidence,
            "azimuth_degrees": azimuth,
            "elevation_degrees": elevation,
            "north_reference": north_reference,
            "target_location": None,
            "target_range_m": None,
            "target_location_observable": False,
        })
        # Magnetic north needs declination before direction can be plotted on
        # geographic (true-north-aligned) maps; don't silently rotate it.
        if not location or north_reference != "true_north" or azimuth is None:
            continue
        display_end = _display_endpoint(location["latitude"], location["longitude"], azimuth)
        if display_end is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [location["longitude"], location["latitude"]],
                    [display_end[1], display_end[0]],
                ],
            },
            "properties": {
                "kind": "bearing_guide", "detection_index": index,
                "object_class": object_class, "confidence": confidence,
                "azimuth_degrees": azimuth, "elevation_degrees": elevation,
                "guide_length_m": _DISPLAY_GUIDE_LENGTH_M,
                "display_only": True,
                "is_target_position": False,
                "is_measured_range": False,
                "north_reference": north_reference,
                "orientation_physically_verified": False,
            },
        })

    return {
        "scene_type": "live_camera_bearing_map",
        "camera_id": camera_id,
        "session_id": session_id,
        "frame_index": frame_index,
        "north_reference": north_reference,
        "calibration_snapshot_sha256": calibration_sha256,
        "session_identity_verified": True,
        "calibration_bound_to_session_verified": True,
        "calibration_bound_to_frame_verified": True,
        "physical_calibration_verified": False,
        "camera_location_bound_to_session_verified": False,
        "camera_location_survey_verified": False,
        "camera_location_source": location["source"] if location else "unavailable",
        "camera_location": location,
        "target_gps_available": False,
        "target_range_available": False,
        "bearing_guides_are_display_only": True,
        "magnetic_declination_applied": False,
        "guides_suppressed_reason": (
            "magnetic_declination_unavailable" if north_reference == "magnetic_north"
            else "camera_location_unavailable" if location is None
            else None
        ),
        "observations": observations,
        "geojson": {"type": "FeatureCollection", "features": features},
    }
