"""Strict input contracts; reject fabricated geo values and untracked UNKNOWNs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
import math
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class CameraCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    source_kind: Literal["USB", "RTSP", "FILE", "SIMULATED", "OTHER"]
    external_ref: str | None = Field(default=None, max_length=160)
    calibration_ref: str | None = Field(default=None, max_length=160)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def coords_pair(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Camera latitude and longitude must be both present or both absent")
        if self.latitude is not None and not self.calibration_ref:
            raise ValueError("Camera location requires traceable calibration_ref")
        return self


class SessionCreate(StrictModel):
    camera_id: str
    source_kind: Literal["LIVE", "VIDEO", "IMAGE", "SIMULATION"]
    validation_level: Literal["SOFTWARE_ONLY", "PHYSICAL_UNVERIFIED"] = "SOFTWARE_ONLY"
    dataset_ref: str | None = Field(default=None, max_length=250)


class GeoInput(StrictModel):
    method: Literal["STEREO_TRIANGULATION", "CALIBRATED_MULTIVIEW", "REGISTERED_GEO", "REFERENCE_MEASUREMENT"]
    evidence_ref: str = Field(min_length=1, max_length=256)
    calibration_ref: str = Field(min_length=1, max_length=160)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    range_m: float | None = Field(default=None, ge=0)
    uncertainty_m: float = Field(ge=0)

    @model_validator(mode="after")
    def source_required(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Geographic coordinates must form a pair")
        if self.latitude is None and self.range_m is None:
            raise ValueError("At least one grounded spatial quantity is required")
        if self.latitude is not None and self.method not in ("REGISTERED_GEO", "REFERENCE_MEASUREMENT", "STEREO_TRIANGULATION", "CALIBRATED_MULTIVIEW"):
            raise ValueError("Unsupported coordinate method")
        return self


class ObjectInput(StrictModel):
    classification: Literal["DRONE", "BIRD", "UNKNOWN"]
    bbox: tuple[float, float, float, float]
    confidence: float | None = Field(default=None, ge=0, le=1)
    track_id: str | None = Field(default=None, min_length=1, max_length=128)
    geo: GeoInput | None = None

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, bbox):
        x1, y1, x2, y2 = bbox
        if not all(math.isfinite(v) for v in bbox) or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be finite, nonnegative XYXY pixel coordinates")
        return bbox

    @model_validator(mode="after")
    def unknown_no_tracking_or_geo(self):
        if self.classification == "UNKNOWN" and (self.track_id is not None or self.geo is not None):
            raise ValueError("UNKNOWN must not have track ID, trajectory or geolocation")
        return self


class FrameIngest(StrictModel):
    session_id: str
    frame_index: int = Field(ge=0)
    captured_at: datetime
    image_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    objects: list[ObjectInput] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def unique_ids_per_frame(self):
        ids = [o.track_id for o in self.objects if o.track_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("One track ID cannot occur twice in one frame")
        return self

    @field_validator("captured_at")
    @classmethod
    def aware_datetime(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at needs a timezone offset (e.g. Z)")
        return value.astimezone(timezone.utc)


class AlertCreate(StrictModel):
    session_id: str
    detection_id: str | None = None
    severity: Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    reason: str = Field(min_length=3, max_length=4000)
    source: str = Field(min_length=1, max_length=60)


class Acknowledge(StrictModel):
    actor: str = Field(min_length=2, max_length=120)


class PerformanceCreate(StrictModel):
    session_id: str
    measured_at: datetime
    fps: float | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    stage: str = Field(min_length=1, max_length=80)
    method: str = Field(min_length=1, max_length=80)

    @field_validator("measured_at")
    @classmethod
    def measured_aware(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("measured_at needs a timezone offset")
        return value.astimezone(timezone.utc)


class EvidenceRegister(StrictModel):
    session_id: str
    relative_path: str = Field(min_length=1, max_length=600)
    media_type: str = Field(min_length=1, max_length=100)


class BenchmarkCreate(StrictModel):
    session_id: str | None = None
    dataset_ref: str = Field(min_length=1, max_length=250)
    source_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    validation_level: Literal["SOFTWARE_ONLY", "PHYSICAL_UNVERIFIED"] = "SOFTWARE_ONLY"
    metrics: dict

    @field_validator("metrics")
    @classmethod
    def metrics_are_json_and_finite(cls, value):
        try:
            encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("metrics must be finite JSON data") from exc
        if len(encoded) > 100_000:
            raise ValueError("metrics payload exceeds 100 KB")
        return value
