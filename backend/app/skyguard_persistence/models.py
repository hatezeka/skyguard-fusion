"""Persistence domain model; intentionally independent from the ML pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def uid() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "sg_cameras"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    external_ref: Mapped[str | None] = mapped_column(String(160))
    calibration_ref: Mapped[str | None] = mapped_column(String(160))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (
        CheckConstraint("source_kind IN ('USB','RTSP','FILE','SIMULATED','OTHER')", name="camera_source"),
        CheckConstraint("(latitude IS NULL AND longitude IS NULL) OR (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180)", name="camera_coords"),
    )


class Session(Base):
    __tablename__ = "sg_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    camera_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_cameras.id", ondelete="RESTRICT"), nullable=False, index=True)
    source_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    validation_level: Mapped[str] = mapped_column(String(24), nullable=False, default="SOFTWARE_ONLY")
    dataset_ref: Mapped[str | None] = mapped_column(String(250))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("source_kind IN ('LIVE','VIDEO','IMAGE','SIMULATION')", name="session_source"),
        CheckConstraint("validation_level IN ('SOFTWARE_ONLY','PHYSICAL_UNVERIFIED','PHYSICAL_VERIFIED')", name="session_level"),
    )


class Frame(Base):
    __tablename__ = "sg_frames"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_sessions.id", ondelete="RESTRICT"), nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    image_sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (
        UniqueConstraint("session_id", "frame_index", name="uq_sg_session_frame"),
        CheckConstraint("frame_index >= 0", name="frame_index_nonnegative"),
        Index("ix_sg_frames_session_time", "session_id", "captured_at"),
    )


class Track(Base):
    __tablename__ = "sg_tracks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_sessions.id", ondelete="RESTRICT"), nullable=False)
    external_track_id: Mapped[str] = mapped_column(String(128), nullable=False)
    classification: Mapped[str] = mapped_column(String(12), nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        UniqueConstraint("session_id", "external_track_id", name="uq_sg_session_track"),
        CheckConstraint("classification IN ('DRONE','BIRD')", name="track_class"),
        Index("ix_sg_tracks_session_class", "session_id", "classification"),
    )


class Detection(Base):
    __tablename__ = "sg_detections"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    frame_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_frames.id", ondelete="RESTRICT"), nullable=False)
    object_index: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(String(12), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    track_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("sg_tracks.id", ondelete="RESTRICT"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (
        UniqueConstraint("frame_id", "object_index", name="uq_sg_frame_object"),
        CheckConstraint("classification IN ('DRONE','BIRD','UNKNOWN')", name="detection_class"),
        CheckConstraint("classification != 'UNKNOWN' OR track_id IS NULL", name="unknown_no_track"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="detection_confidence"),
        CheckConstraint("x2 > x1 AND y2 > y1 AND x1 >= 0 AND y1 >= 0", name="detection_bbox"),
        Index("ix_sg_detection_track", "track_id"),
    )


class TrackObservation(Base):
    __tablename__ = "sg_track_observations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    track_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_tracks.id", ondelete="RESTRICT"), nullable=False)
    detection_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_detections.id", ondelete="RESTRICT"), nullable=False, unique=True)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    __table_args__ = (Index("ix_sg_obs_track_time", "track_id", "observed_at"),)


class GeoEvidence(Base):
    __tablename__ = "sg_geo_evidence"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    detection_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_detections.id", ondelete="RESTRICT"), nullable=False, unique=True)
    method: Mapped[str] = mapped_column(String(80), nullable=False)
    evidence_ref: Mapped[str] = mapped_column(String(256), nullable=False)
    calibration_ref: Mapped[str | None] = mapped_column(String(160))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    range_m: Mapped[float | None] = mapped_column(Float)
    uncertainty_m: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (
        CheckConstraint("(latitude IS NULL AND longitude IS NULL) OR (latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180)", name="geo_coordinates"),
        CheckConstraint("range_m IS NULL OR range_m >= 0", name="geo_positive_range"),
        CheckConstraint("uncertainty_m >= 0", name="geo_positive_uncertainty"),
        CheckConstraint("length(evidence_ref) > 0 AND length(method) > 0", name="geo_provenance"),
    )


class Alert(Base):
    __tablename__ = "sg_alerts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_sessions.id", ondelete="RESTRICT"), nullable=False, index=True)
    detection_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("sg_detections.id", ondelete="RESTRICT"))
    severity: Mapped[str] = mapped_column(String(12), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(60), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")
    acknowledged_by: Mapped[str | None] = mapped_column(String(120))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (
        CheckConstraint("severity IN ('INFO','LOW','MEDIUM','HIGH','CRITICAL')", name="alert_severity"),
        CheckConstraint("status IN ('OPEN','ACKNOWLEDGED')", name="alert_status"),
    )


class PerformanceSample(Base):
    __tablename__ = "sg_performance_samples"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_sessions.id", ondelete="RESTRICT"), nullable=False, index=True)
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fps: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    stage: Mapped[str] = mapped_column(String(80), nullable=False)
    method: Mapped[str] = mapped_column(String(80), nullable=False)
    __table_args__ = (CheckConstraint("fps IS NULL OR fps >= 0", name="fps_nonnegative"), CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_nonnegative"))


class EvidenceFile(Base):
    __tablename__ = "sg_evidence_files"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("sg_sessions.id", ondelete="RESTRICT"), nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(String(600), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (UniqueConstraint("session_id", "relative_path", name="uq_sg_evidence_path"), CheckConstraint("size_bytes >= 0", name="evidence_size"))


class BenchmarkRun(Base):
    __tablename__ = "sg_benchmark_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    session_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("sg_sessions.id", ondelete="RESTRICT"), index=True)
    dataset_ref: Mapped[str] = mapped_column(String(250), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    validation_level: Mapped[str] = mapped_column(String(24), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (CheckConstraint("validation_level IN ('SOFTWARE_ONLY','PHYSICAL_UNVERIFIED','PHYSICAL_VERIFIED')", name="benchmark_level"),)


class AuditEvent(Base):
    __tablename__ = "sg_audit_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    actor: Mapped[str] = mapped_column(String(120), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    __table_args__ = (Index("ix_sg_audit_created", "created_at"),)
