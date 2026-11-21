"""Black-box contracts for persistent detection, tracking, spatial evidence and API auth."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.skyguard_persistence.api import create_router
from app.skyguard_persistence.database import Database
from app.skyguard_persistence.models import Detection
from app.skyguard_persistence.schemas import (Acknowledge, AlertCreate, BenchmarkCreate,
    CameraCreate, EvidenceRegister, FrameIngest, PerformanceCreate, SessionCreate)
from app.skyguard_persistence.service import Conflict, InvalidOperation, Store


@pytest.fixture
def store(tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'test.sqlite3').as_posix()}")
    db.create_schema_for_tests_only()
    yield Store(db)
    db.dispose()


def base_session(store):
    cam = store.create_camera(CameraCreate(name="Camera 01", source_kind="FILE"))
    ses = store.create_session(SessionCreate(camera_id=cam["id"], source_kind="VIDEO"))
    return cam, ses


def frame(sid, idx=0, objs=None):
    return FrameIngest(session_id=sid, frame_index=idx, captured_at=f"2026-11-20T12:00:{idx:02d}Z",
                       objects=objs if objs is not None else [
                           {"classification": "BIRD", "bbox": [1, 1, 20, 30], "confidence": 0.9, "track_id": "bird-1"}
                       ])


def test_db_foreign_keys_enforced(store):
    with pytest.raises(IntegrityError):
        with store.db.sessionmaker.begin() as tx:
            tx.add(Detection(frame_id="missing", object_index=0, classification="UNKNOWN", x1=0, y1=0, x2=1, y2=1))


def test_happy_path_tracks_observations_and_summary(store):
    _, ses = base_session(store)
    assert store.ingest_frame(frame(ses["id"]))["created"] is True
    store.ingest_frame(frame(ses["id"], 1))
    summary = store.session_summary(ses["id"])
    assert summary["frames"] == 2
    assert summary["tracks"] == 1
    assert summary["detections"] == {"BIRD": 2}
    tracks = store.list_tracks(ses["id"])
    assert len(store.track_history(tracks[0]["id"])) == 2


def test_replay_is_idempotent(store):
    _, ses = base_session(store)
    first = store.ingest_frame(frame(ses["id"]))
    second = store.ingest_frame(frame(ses["id"]))
    assert first["frame_id"] == second["frame_id"]
    assert not second["created"]
    assert len(store.list_detections(ses["id"])) == 1


def test_conflicting_frame_not_overwritten(store):
    _, ses = base_session(store)
    store.ingest_frame(frame(ses["id"]))
    with pytest.raises(Conflict) as exc:
        store.ingest_frame(frame(ses["id"], objs=[{"classification": "UNKNOWN", "bbox": [1, 1, 2, 2]}]))
    assert exc.value.status_code == 409
    assert store.session_summary(ses["id"])["frames"] == 1


def test_unknown_is_logged_but_never_tracked(store):
    _, ses = base_session(store)
    store.ingest_frame(frame(ses["id"], objs=[{"classification": "UNKNOWN", "bbox": [1, 1, 20, 30]}]))
    assert store.list_tracks(ses["id"]) == []
    assert store.list_detections(ses["id"])[0]["track_id"] is None
    assert store.list_geo(ses["id"]) == []


@pytest.mark.parametrize("extra", [
    {"track_id": "secret"},
    {"geo": {"method": "REGISTERED_GEO", "evidence_ref": "x", "calibration_ref": "y",
             "latitude": 42, "longitude": 21, "uncertainty_m": 2}},
])
def test_unknown_with_track_or_geo_rejected(extra):
    with pytest.raises(ValidationError):
        FrameIngest(session_id="x", frame_index=0, captured_at="2026-11-20T00:00:00Z",
                    objects=[dict(classification="UNKNOWN", bbox=[0, 0, 1, 1], **extra)])


def test_track_class_switch_rolls_back_entire_frame(store):
    _, ses = base_session(store)
    store.ingest_frame(frame(ses["id"]))
    with pytest.raises(Conflict):
        store.ingest_frame(frame(ses["id"], 1, objs=[
            {"classification": "DRONE", "bbox": [2, 2, 3, 3], "track_id": "bird-1"}]))
    assert store.session_summary(ses["id"])["frames"] == 1


def test_invalid_bbox_and_timestamps_rejected():
    with pytest.raises(ValidationError):
        FrameIngest(session_id="x", frame_index=0, captured_at="2026-11-20T00:00:00",
                    objects=[])
    with pytest.raises(ValidationError):
        FrameIngest(session_id="x", frame_index=0, captured_at="2026-11-20T00:00:00Z",
                    objects=[{"classification": "BIRD", "bbox": [0, 0, 0, 9]}])
    with pytest.raises(ValidationError):
        FrameIngest(session_id="x", frame_index=0, captured_at="2026-11-20T00:00:00Z",
                    objects=[{"classification": "BIRD", "bbox": [0, 0, float("nan"), 9]}])


def test_duplicate_track_id_same_frame_rejected():
    with pytest.raises(ValidationError):
        frame("x", objs=[
            {"classification": "BIRD", "bbox": [0, 0, 2, 2], "track_id": "x"},
            {"classification": "BIRD", "bbox": [2, 0, 4, 2], "track_id": "x"},
        ])


def test_geospatial_evidence_requires_reference_and_uncertainty(store):
    _, ses = base_session(store)
    with pytest.raises(ValidationError):
        frame(ses["id"], objs=[{"classification": "DRONE", "bbox": [0, 0, 9, 9],
            "geo": {"method": "REGISTERED_GEO", "latitude": 42.8, "longitude": 20.9, "uncertainty_m": 2}}])
    payload = frame(ses["id"], objs=[{"classification": "DRONE", "bbox": [0, 0, 9, 9],
        "track_id": "drone-1", "geo": {"method": "REGISTERED_GEO", "latitude": 42.8,
        "longitude": 20.9, "uncertainty_m": 3.0, "evidence_ref": "survey/a1",
        "calibration_ref": "calib/2026"}}])
    store.ingest_frame(payload)
    got = store.list_geo(ses["id"])
    assert len(got) == 1
    assert got[0]["uncertainty_m"] == 3.0 and got[0]["evidence_ref"] == "survey/a1"


def test_camera_coordinates_need_calibration():
    with pytest.raises(ValidationError):
        CameraCreate(name="camera", source_kind="USB", latitude=42, longitude=21)


def test_alert_ack_and_audit(store):
    _, ses = base_session(store)
    a = store.create_alert(AlertCreate(session_id=ses["id"], severity="HIGH",
                                      reason="Validated risk evidence", source="risk_engine"))
    ack = store.ack_alert(a["id"], Acknowledge(actor="operator-1"))
    assert ack["status"] == "ACKNOWLEDGED"
    assert ack["acknowledged_by"] == "operator-1"
    assert store.ack_alert(a["id"], Acknowledge(actor="operator-2"))["acknowledged_by"] == "operator-1"
    assert any(x["action"] == "alert.acknowledge" for x in store.list_audit())


def test_stop_session_prevents_new_frames(store):
    _, ses = base_session(store)
    store.close_session(ses["id"])
    with pytest.raises(Conflict):
        store.ingest_frame(frame(ses["id"]))


def test_metrics_benchmarks_never_promote_unverified_to_physical(store):
    _, ses = base_session(store)
    metric = store.record_performance(PerformanceCreate(session_id=ses["id"],
        measured_at="2026-11-20T12:00:00Z", fps=22.5, latency_ms=12.0,
        stage="end_to_end", method="perf_counter"))
    assert metric["fps"] == 22.5
    assert len(store.list_performance(ses["id"])) == 1
    with pytest.raises(ValidationError):
        BenchmarkCreate(dataset_ref="test", source_sha256="a"*64,
                        validation_level="PHYSICAL_VERIFIED", metrics={})
    bench = store.record_benchmark(BenchmarkCreate(session_id=ses["id"], dataset_ref="test",
        source_sha256="a"*64, metrics={"idf1": 0.73}))
    assert bench["validation_level"] == "SOFTWARE_ONLY"
    assert store.list_benchmarks()[0]["metrics"]["idf1"] == 0.73


def test_evidence_path_hash_and_immutable_on_change(store, tmp_path):
    _, ses = base_session(store)
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "shot.png").write_bytes(b"camera frame original")
    record = store.register_evidence(EvidenceRegister(session_id=ses["id"],
                                       relative_path="shot.png", media_type="image/png"), root)
    assert record["sha256"] == hashlib.sha256(b"camera frame original").hexdigest()
    assert len(store.list_evidence(ses["id"])) == 1
    with pytest.raises(InvalidOperation):
        store.register_evidence(EvidenceRegister(session_id=ses["id"],
                                relative_path="../outside", media_type="image/png"), root)
    (root / "shot.png").write_bytes(b"tampered")
    with pytest.raises(Conflict):
        store.register_evidence(EvidenceRegister(session_id=ses["id"],
                                       relative_path="shot.png", media_type="image/png"), root)


def test_protect_database_from_unknown_track_sql(store):
    _, ses = base_session(store)
    store.ingest_frame(frame(ses["id"]))
    with pytest.raises(IntegrityError):
        with store.db.sessionmaker.begin() as tx:
            det = tx.get(Detection, store.list_detections(ses["id"])[0]["id"])
            det.classification = "UNKNOWN"
            tx.flush()


def test_http_auth_and_endpoints(store):
    app = FastAPI()
    app.include_router(create_router(store.db, api_key="unit-test-secret"))
    client = TestClient(app)
    assert client.get("/api/persistence/cameras").status_code == 401
    assert client.get("/api/persistence/cameras", headers={"Authorization": "Bearer wrong"}).status_code == 403
    h = {"Authorization": "Bearer unit-test-secret"}
    assert client.get("/api/persistence/health", headers=h).status_code == 200
    camera = client.post("/api/persistence/cameras", json={"name": "C", "source_kind": "USB"}, headers=h)
    assert camera.status_code == 201, camera.text
    session = client.post("/api/persistence/sessions", json={"camera_id": camera.json()["id"], "source_kind": "LIVE"}, headers=h)
    assert session.status_code == 201, session.text
    data = {"session_id": session.json()["id"], "frame_index": 0,
            "captured_at": "2026-11-20T12:00:00Z",
            "objects": [{"classification": "UNKNOWN", "bbox": [0,0,10,10]}]}
    assert client.post("/api/persistence/frames", json=data, headers=h).status_code == 201
    assert client.post("/api/persistence/frames", json=data, headers=h).json()["created"] is False
    assert client.get("/api/persistence/sessions/"+session.json()["id"]+"/summary", headers=h).json()["detections"] == {"UNKNOWN": 1}
    assert client.get("/api/persistence/detections", headers=h).json()[0]["classification"] == "UNKNOWN"
    assert client.get("/api/persistence/tracks", headers=h).json() == []
    assert client.get("/api/persistence/detections?limit=9999", headers=h).status_code == 422
    assert client.get("/api/persistence/audit", headers=h).status_code == 200


def test_api_fails_closed_without_key(store, monkeypatch):
    monkeypatch.delenv("SKYGUARD_DB_API_KEY", raising=False)
    app = FastAPI()
    app.include_router(create_router(store.db))
    assert TestClient(app).get("/api/persistence/health").status_code == 503


def test_alembic_upgrade_and_schema_replay(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    url = f"sqlite:///{(tmp_path/'migrated.sqlite3').as_posix()}"
    env = dict(os.environ, SKYGUARD_DATABASE_URL=url)
    for _ in range(2):
        subprocess.run([sys.executable, "-m", "alembic", "-c", "skyguard_db.ini", "upgrade", "head"],
                       cwd=backend, env=env, check=True, capture_output=True, text=True)
    db = Database(url)
    with db.engine.connect() as con:
        tables = {row[0] for row in con.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        assert {"sg_cameras", "sg_sessions", "sg_detections", "sg_tracks", "sg_audit_events", "alembic_version"}.issubset(tables)
        # SKYGUARD_MIGRATION_HEAD_TEST_V4: verify actual Alembic head and the 0002 media schema.
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        alembic_config = Config(str(backend / "skyguard_db.ini"))
        current_head = ScriptDirectory.from_config(alembic_config).get_current_head()
        assert current_head is not None
        assert con.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == current_head
        assert "sg_media_jobs" in tables
        media_columns = {row[1] for row in con.execute(text('PRAGMA table_info("sg_media_jobs")'))}
        assert {"id", "kind", "status", "source_sha256", "source_file", "session_id",
                "frames_processed", "summary", "created_at", "updated_at"}.issubset(media_columns)
        media_indexes = {row[1] for row in con.execute(text('PRAGMA index_list("sg_media_jobs")'))}
        assert "ix_sg_media_jobs_created_at" in media_indexes
        media_foreign_keys = con.execute(text('PRAGMA foreign_key_list("sg_media_jobs")')).fetchall()
        assert any(row[2] == "sg_sessions" and row[3] == "session_id" for row in media_foreign_keys)
        assert con.execute(text("SELECT COUNT(*) FROM sg_media_jobs")).scalar_one() == 0
    db.dispose()


def test_benchmark_metrics_reject_non_finite_json():
    with pytest.raises(ValidationError):
        BenchmarkCreate(dataset_ref="sample", source_sha256="b" * 64,
                        metrics={"fps": float("nan")})


def test_sqlite_foreign_key_integrity_after_migration(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "migrated.sqlite3"
    env = dict(os.environ, SKYGUARD_DATABASE_URL=f"sqlite:///{db_path.as_posix()}")
    subprocess.run([sys.executable, "-m", "alembic", "-c", "skyguard_db.ini", "upgrade", "head"],
                   cwd=backend, env=env, check=True, capture_output=True, text=True)
    db = Database(env["SKYGUARD_DATABASE_URL"])
    with db.engine.connect() as con:
        assert con.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        assert con.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
        assert con.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
    db.dispose()
