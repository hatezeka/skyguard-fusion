"""Transactional persistence. No inference, geometry or classification is fabricated here."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from .database import Database
from .models import (Alert, AuditEvent, BenchmarkRun, Camera, Detection, EvidenceFile,
                     Frame, GeoEvidence, PerformanceSample, Session, Track, TrackObservation, utcnow)
from .schemas import (Acknowledge, AlertCreate, BenchmarkCreate, CameraCreate,
                      EvidenceRegister, FrameIngest, PerformanceCreate, SessionCreate)


class Conflict(HTTPException):
    def __init__(self, detail="Conflict"):
        super().__init__(status_code=409, detail=detail)


class NotFound(HTTPException):
    def __init__(self, detail="Not found"):
        super().__init__(status_code=404, detail=detail)


class InvalidOperation(HTTPException):
    def __init__(self, detail="Invalid operation"):
        super().__init__(status_code=422, detail=detail)


def serialize(obj) -> dict:
    from sqlalchemy import inspect
    result = {}
    for attr in inspect(obj).mapper.column_attrs:
        value = getattr(obj, attr.key)
        if isinstance(value, datetime):
            # SQLite may return naive datetimes; these were always stored as UTC.
            value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
            value = value.isoformat().replace("+00:00", "Z")
        result[attr.key] = value
    return result


def audit(tx, actor: str, action: str, entity_type: str, entity_id: str, details: dict | None = None):
    tx.add(AuditEvent(actor=actor, action=action, entity_type=entity_type,
                      entity_id=entity_id, details=details or {}))


def get_or_raise(tx, model, entity_id):
    result = tx.get(model, entity_id)
    if result is None:
        raise NotFound(f"{model.__name__} not found")
    return result


class Store:
    def __init__(self, database: Database):
        self.db = database

    def create_camera(self, item: CameraCreate, actor="backend") -> dict:
        with self.db.sessionmaker.begin() as tx:
            model = Camera(**item.model_dump())
            tx.add(model)
            tx.flush()
            audit(tx, actor, "camera.create", "camera", model.id)
            return serialize(model)

    def list_cameras(self, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            rows = tx.scalars(select(Camera).order_by(Camera.created_at.desc(), Camera.id).limit(limit)).all()
            return [serialize(row) for row in rows]

    def create_session(self, item: SessionCreate, actor="backend") -> dict:
        with self.db.sessionmaker.begin() as tx:
            get_or_raise(tx, Camera, item.camera_id)
            model = Session(**item.model_dump())
            tx.add(model)
            tx.flush()
            audit(tx, actor, "session.create", "session", model.id, {"validation_level": model.validation_level})
            return serialize(model)

    def close_session(self, session_id: str, actor="backend") -> dict:
        with self.db.sessionmaker.begin() as tx:
            row = get_or_raise(tx, Session, session_id)
            if row.ended_at is None:
                row.ended_at = utcnow()
                audit(tx, actor, "session.close", "session", session_id)
            tx.flush()
            return serialize(row)

    def list_sessions(self, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            return [serialize(row) for row in tx.scalars(select(Session).order_by(Session.started_at.desc()).limit(limit)).all()]

    def ingest_frame(self, item: FrameIngest, actor="pipeline") -> dict:
        canonical = json.dumps(item.model_dump(mode="json", exclude_none=True), sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        try:
            with self.db.sessionmaker.begin() as tx:
                session = get_or_raise(tx, Session, item.session_id)
                existing = tx.scalar(select(Frame).where(Frame.session_id == item.session_id,
                                                         Frame.frame_index == item.frame_index))
                if existing:
                    if existing.payload_sha256 != digest:
                        raise Conflict("Frame index already stored with different content")
                    return {"frame_id": existing.id, "created": False, "detections": len(item.objects)}
                if session.ended_at is not None:
                    raise Conflict("Monitoring session has already ended")
                frame = Frame(session_id=item.session_id, frame_index=item.frame_index,
                              captured_at=item.captured_at, image_sha256=item.image_sha256,
                              payload_sha256=digest)
                tx.add(frame)
                tx.flush()
                for index, item_obj in enumerate(item.objects):
                    track = None
                    if item_obj.track_id is not None:
                        track = tx.scalar(select(Track).where(Track.session_id == item.session_id,
                                                             Track.external_track_id == item_obj.track_id))
                        if track is None:
                            track = Track(session_id=item.session_id, external_track_id=item_obj.track_id,
                                          classification=item_obj.classification,
                                          first_seen=item.captured_at, last_seen=item.captured_at)
                            tx.add(track)
                            tx.flush()
                        elif track.classification != item_obj.classification:
                            raise Conflict("Track ID may not switch DRONE/BIRD classification within a session")
                        else:
                            first = track.first_seen.replace(tzinfo=timezone.utc) if track.first_seen.tzinfo is None else track.first_seen
                            last = track.last_seen.replace(tzinfo=timezone.utc) if track.last_seen.tzinfo is None else track.last_seen
                            if item.captured_at < first:
                                track.first_seen = item.captured_at
                            if item.captured_at > last:
                                track.last_seen = item.captured_at
                    det = Detection(frame_id=frame.id, object_index=index,
                                    classification=item_obj.classification, confidence=item_obj.confidence,
                                    x1=item_obj.bbox[0], y1=item_obj.bbox[1],
                                    x2=item_obj.bbox[2], y2=item_obj.bbox[3],
                                    track_id=track.id if track else None)
                    tx.add(det)
                    tx.flush()
                    if track:
                        tx.add(TrackObservation(track_id=track.id, detection_id=det.id,
                                                frame_index=item.frame_index, observed_at=item.captured_at))
                    if item_obj.geo is not None:
                        tx.add(GeoEvidence(detection_id=det.id, **item_obj.geo.model_dump()))
                audit(tx, actor, "frame.ingested", "frame", frame.id,
                      {"session_id": item.session_id, "frame_index": item.frame_index,
                       "objects": len(item.objects), "sha256": digest})
                return {"frame_id": frame.id, "created": True, "detections": len(item.objects)}
        except IntegrityError as exc:
            # A concurrent retry of the same frame can violate unique(session, index).
            with self.db.sessionmaker() as tx:
                old = tx.scalar(select(Frame).where(Frame.session_id == item.session_id,
                                                    Frame.frame_index == item.frame_index))
                if old and old.payload_sha256 == digest:
                    return {"frame_id": old.id, "created": False, "detections": len(item.objects)}
            raise Conflict("Concurrent frame conflict or invalid reference") from exc

    def list_detections(self, session_id: str | None = None, classification: str | None = None,
                        limit=100, offset=0) -> list[dict]:
        with self.db.sessionmaker() as tx:
            statement = (select(Detection, Frame.session_id, Frame.frame_index, Frame.captured_at)
                         .join(Frame, Frame.id == Detection.frame_id))
            if session_id:
                statement = statement.where(Frame.session_id == session_id)
            if classification:
                statement = statement.where(Detection.classification == classification)
            rows = tx.execute(statement.order_by(Frame.captured_at.desc(), Frame.frame_index.desc(),
                                                  Detection.object_index).limit(limit).offset(offset)).all()
            return [dict(serialize(det), session_id=sid, frame_index=idx,
                         captured_at=serialize_timestamp(at)) for det, sid, idx, at in rows]

    def list_tracks(self, session_id: str | None = None, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            q = select(Track)
            if session_id:
                q = q.where(Track.session_id == session_id)
            return [serialize(r) for r in tx.scalars(q.order_by(Track.last_seen.desc()).limit(limit)).all()]

    def track_history(self, track_id: str, limit=200) -> list[dict]:
        with self.db.sessionmaker() as tx:
            get_or_raise(tx, Track, track_id)
            statement = (select(TrackObservation, Detection)
                         .join(Detection, Detection.id == TrackObservation.detection_id)
                         .where(TrackObservation.track_id == track_id)
                         .order_by(TrackObservation.frame_index.asc()).limit(limit))
            return [{"observation": serialize(obs), "detection": serialize(det)}
                    for obs, det in tx.execute(statement)]

    def list_geo(self, session_id: str | None = None, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            stmt = (select(GeoEvidence, Detection.classification, Frame.session_id)
                    .join(Detection, Detection.id == GeoEvidence.detection_id)
                    .join(Frame, Frame.id == Detection.frame_id))
            if session_id:
                stmt = stmt.where(Frame.session_id == session_id)
            return [dict(serialize(geo), classification=cls, session_id=sid)
                    for geo, cls, sid in tx.execute(stmt.order_by(GeoEvidence.created_at.desc()).limit(limit))]

    def create_alert(self, item: AlertCreate, actor="guardian") -> dict:
        with self.db.sessionmaker.begin() as tx:
            get_or_raise(tx, Session, item.session_id)
            if item.detection_id:
                det = get_or_raise(tx, Detection, item.detection_id)
                frame = get_or_raise(tx, Frame, det.frame_id)
                if frame.session_id != item.session_id:
                    raise InvalidOperation("Detection belongs to another monitoring session")
            row = Alert(**item.model_dump())
            tx.add(row)
            tx.flush()
            audit(tx, actor, "alert.create", "alert", row.id, {"severity": row.severity})
            return serialize(row)

    def ack_alert(self, alert_id: str, item: Acknowledge) -> dict:
        with self.db.sessionmaker.begin() as tx:
            row = get_or_raise(tx, Alert, alert_id)
            if row.status != "ACKNOWLEDGED":
                row.status = "ACKNOWLEDGED"
                row.acknowledged_by = item.actor
                row.acknowledged_at = utcnow()
                audit(tx, item.actor, "alert.acknowledge", "alert", row.id)
            tx.flush()
            return serialize(row)

    def list_alerts(self, session_id: str | None = None, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            q = select(Alert)
            if session_id:
                q = q.where(Alert.session_id == session_id)
            return [serialize(a) for a in tx.scalars(q.order_by(Alert.created_at.desc()).limit(limit)).all()]

    def record_performance(self, item: PerformanceCreate, actor="pipeline") -> dict:
        with self.db.sessionmaker.begin() as tx:
            get_or_raise(tx, Session, item.session_id)
            row = PerformanceSample(**item.model_dump())
            tx.add(row)
            tx.flush()
            audit(tx, actor, "performance.record", "performance", row.id)
            return serialize(row)

    def list_performance(self, session_id: str, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            return [serialize(row) for row in tx.scalars(select(PerformanceSample)
                    .where(PerformanceSample.session_id == session_id)
                    .order_by(PerformanceSample.measured_at.desc()).limit(limit)).all()]

    def register_evidence(self, item: EvidenceRegister, evidence_root: Path, actor="backend") -> dict:
        root = evidence_root.resolve(strict=True)
        relative = Path(item.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise InvalidOperation("Unsafe evidence path")
        try:
            path = (root / relative).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InvalidOperation("Evidence file does not exist or cannot be resolved") from exc
        if not path.is_relative_to(root) or not path.is_file():
            raise InvalidOperation("Evidence path must be a real file inside evidence root")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        with self.db.sessionmaker.begin() as tx:
            get_or_raise(tx, Session, item.session_id)
            exists = tx.scalar(select(EvidenceFile).where(EvidenceFile.session_id == item.session_id,
                                                         EvidenceFile.relative_path == relative.as_posix()))
            if exists:
                if exists.sha256 != digest.hexdigest() or exists.size_bytes != path.stat().st_size:
                    raise Conflict("Existing evidence path has changed content")
                return serialize(exists)
            row = EvidenceFile(session_id=item.session_id, relative_path=relative.as_posix(),
                               sha256=digest.hexdigest(), size_bytes=path.stat().st_size,
                               media_type=item.media_type)
            tx.add(row)
            tx.flush()
            audit(tx, actor, "evidence.register", "evidence", row.id, {"sha256": row.sha256})
            return serialize(row)

    def list_evidence(self, session_id: str, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            return [serialize(row) for row in tx.scalars(select(EvidenceFile)
                    .where(EvidenceFile.session_id == session_id)
                    .order_by(EvidenceFile.created_at.desc()).limit(limit)).all()]

    def record_benchmark(self, item: BenchmarkCreate, actor="evaluator") -> dict:
        with self.db.sessionmaker.begin() as tx:
            if item.session_id:
                get_or_raise(tx, Session, item.session_id)
            row = BenchmarkRun(**item.model_dump())
            tx.add(row)
            tx.flush()
            audit(tx, actor, "benchmark.record", "benchmark", row.id,
                  {"validation_level": row.validation_level})
            return serialize(row)

    def list_benchmarks(self, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            return [serialize(row) for row in tx.scalars(select(BenchmarkRun)
                    .order_by(BenchmarkRun.created_at.desc()).limit(limit)).all()]

    def list_audit(self, limit=100) -> list[dict]:
        with self.db.sessionmaker() as tx:
            return [serialize(row) for row in tx.scalars(select(AuditEvent)
                    .order_by(AuditEvent.created_at.desc()).limit(limit)).all()]

    def session_summary(self, session_id: str) -> dict:
        with self.db.sessionmaker() as tx:
            get_or_raise(tx, Session, session_id)
            breakdown = tx.execute(select(Detection.classification, func.count(Detection.id))
                                   .join(Frame, Frame.id == Detection.frame_id)
                                   .where(Frame.session_id == session_id)
                                   .group_by(Detection.classification)).all()
            frames = tx.scalar(select(func.count(Frame.id)).where(Frame.session_id == session_id))
            tracks = tx.scalar(select(func.count(Track.id)).where(Track.session_id == session_id))
            alerts = tx.scalar(select(func.count(Alert.id)).where(Alert.session_id == session_id))
            return {"session_id": session_id, "frames": frames, "tracks": tracks, "alerts": alerts,
                    "detections": {name: count for name, count in breakdown},
                    "note": "Counts are observations, not independently verified classification accuracy"}


def serialize_timestamp(value: datetime) -> str:
    return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)).isoformat().replace("+00:00", "Z")
