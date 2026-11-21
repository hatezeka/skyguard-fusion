"""Opt-in, bearer-protected persistence endpoints. Mount under existing auth or use standalone."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text

from .database import Database
from .schemas import (Acknowledge, AlertCreate, BenchmarkCreate, CameraCreate,
                      EvidenceRegister, FrameIngest, PerformanceCreate, SessionCreate)
from .service import Store


def create_router(db: Database | None = None, *, api_key: str | None = None,
                  evidence_root: str | Path | None = None) -> APIRouter:
    database = db or Database()
    store = Store(database)
    token = api_key if api_key is not None else os.getenv("SKYGUARD_DB_API_KEY", "")
    evidence_dir = Path(evidence_root or os.getenv("SKYGUARD_EVIDENCE_ROOT", "")).expanduser() if (evidence_root or os.getenv("SKYGUARD_EVIDENCE_ROOT")) else None
    router = APIRouter(prefix="/api/persistence", tags=["SkyGuard evidence & persistence"])

    security = HTTPBearer(auto_error=False)

    def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
        if not token:
            raise HTTPException(status_code=503, detail="Persistence API disabled until SKYGUARD_DB_API_KEY is configured")
        if credentials is None:
            raise HTTPException(status_code=401, detail="Bearer token required", headers={"WWW-Authenticate": "Bearer"})
        supplied = credentials.credentials
        if not secrets.compare_digest(supplied.encode("utf8"), token.encode("utf8")):
            raise HTTPException(status_code=403, detail="Invalid access token")

    secure = [Depends(require_auth)]

    @router.get("/health", dependencies=secure)
    def health():
        with database.engine.connect() as con:
            con.execute(text("SELECT 1"))
        return {"status": "ok", "backend": database.engine.dialect.name}

    @router.post("/cameras", status_code=201, dependencies=secure)
    def create_camera(item: CameraCreate):
        return store.create_camera(item)

    @router.get("/cameras", dependencies=secure)
    def list_cameras(limit: int = Query(100, ge=1, le=200)):
        return store.list_cameras(limit)

    @router.post("/sessions", status_code=201, dependencies=secure)
    def create_session(item: SessionCreate):
        return store.create_session(item)

    @router.get("/sessions", dependencies=secure)
    def list_sessions(limit: int = Query(100, ge=1, le=200)):
        return store.list_sessions(limit)

    @router.post("/sessions/{session_id}/close", dependencies=secure)
    def close_session(session_id: str):
        return store.close_session(session_id)

    @router.get("/sessions/{session_id}/summary", dependencies=secure)
    def session_summary(session_id: str):
        return store.session_summary(session_id)

    @router.post("/frames", status_code=201, dependencies=secure)
    def ingest_frame(item: FrameIngest):
        return store.ingest_frame(item)

    @router.get("/detections", dependencies=secure)
    def list_detections(session_id: str | None = None,
                        classification: str | None = Query(None, pattern="^(DRONE|BIRD|UNKNOWN)$"),
                        limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0, le=100000)):
        return store.list_detections(session_id, classification, limit, offset)

    @router.get("/tracks", dependencies=secure)
    def list_tracks(session_id: str | None = None, limit: int = Query(100, ge=1, le=200)):
        return store.list_tracks(session_id, limit)

    @router.get("/tracks/{track_id}/history", dependencies=secure)
    def track_history(track_id: str, limit: int = Query(200, ge=1, le=500)):
        return store.track_history(track_id, limit)

    @router.get("/geospatial", dependencies=secure)
    def get_geo(session_id: str | None = None, limit: int = Query(100, ge=1, le=200)):
        return store.list_geo(session_id, limit)

    @router.post("/alerts", status_code=201, dependencies=secure)
    def create_alert(item: AlertCreate):
        return store.create_alert(item)

    @router.post("/alerts/{alert_id}/acknowledge", dependencies=secure)
    def acknowledge(alert_id: str, item: Acknowledge):
        return store.ack_alert(alert_id, item)

    @router.get("/alerts", dependencies=secure)
    def list_alerts(session_id: str | None = None, limit: int = Query(100, ge=1, le=200)):
        return store.list_alerts(session_id, limit)

    @router.post("/performance", status_code=201, dependencies=secure)
    def record_performance(item: PerformanceCreate):
        return store.record_performance(item)

    @router.get("/performance", dependencies=secure)
    def list_performance(session_id: str, limit: int = Query(100, ge=1, le=200)):
        return store.list_performance(session_id, limit)

    @router.post("/evidence", status_code=201, dependencies=secure)
    def register_evidence(item: EvidenceRegister):
        if evidence_dir is None:
            raise HTTPException(status_code=503, detail="SKYGUARD_EVIDENCE_ROOT not configured")
        if not evidence_dir.is_dir():
            raise HTTPException(status_code=503, detail="Evidence directory unavailable")
        return store.register_evidence(item, evidence_dir)

    @router.get("/evidence", dependencies=secure)
    def list_evidence(session_id: str, limit: int = Query(100, ge=1, le=200)):
        return store.list_evidence(session_id, limit)

    @router.post("/benchmarks", status_code=201, dependencies=secure)
    def create_benchmark(item: BenchmarkCreate):
        return store.record_benchmark(item)

    @router.get("/benchmarks", dependencies=secure)
    def list_benchmarks(limit: int = Query(100, ge=1, le=200)):
        return store.list_benchmarks(limit)

    @router.get("/audit", dependencies=secure)
    def list_audit(limit: int = Query(100, ge=1, le=200)):
        return store.list_audit(limit)

    return router
