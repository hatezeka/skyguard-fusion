"""Real SQLite I/O / restart tests. Synthetic input is TEST DATA only.

No real model accuracy or physical field performance is claimed here.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, func

from app.skyguard_persistence.database import Database
from app.skyguard_persistence.models import Camera, Frame, Session, Detection, Track, GeoEvidence
from app.skyguard_persistence.live_capture import LiveRecorder, capture_payload
from app.skyguard_persistence.api import create_router


@dataclass
class Box:
    x: float = 2
    y: float = 3
    width: float = 10
    height: float = 12


@dataclass
class DetectionInput:
    object_class: str
    confidence: float = 0.7
    bbox: Box = None

    def __post_init__(self):
        if self.bbox is None:
            self.bbox = Box()


class FakeManager:
    def __init__(self, state='running'):
        self._state = state
    def status(self):
        return SimpleNamespace(state=self._state)


def snapshot(sid='live-session-1', index=0):
    return SimpleNamespace(state='running', session_id=sid, camera_id=None,
                           frame_index=index, processing_seconds=0.012)


def capture(rec, frame=0, sid='live-session-1'):
    drone = DetectionInput('drone')
    unknown = DetectionInput('unknown', bbox=Box(30, 4, 9, 7))
    track = SimpleNamespace(detection=drone, track_id=17, frame_index=frame, state='confirmed')
    rec.capture(snapshot=snapshot(sid, frame), raw_detections=[drone, unknown],
                confirmed_tracks=[track], source_type='usb', source_location=0, jpeg_bytes=b'image-pixels')


def test_direct_contract_never_fabricates_ids_or_gps():
    drone = DetectionInput('drone')
    unknown = DetectionInput('helicopter', bbox=Box(20, 20, 4, 4))
    matching = SimpleNamespace(detection=drone, track_id=123, frame_index=1, state='confirmed')
    event = capture_payload(snapshot(index=1), [drone, unknown], [matching], 'usb', source_location=0,
                            clock=lambda: datetime(2026, 11, 21, tzinfo=timezone.utc))
    assert event.objects[0].track_id == '123'
    assert event.objects[1].classification == 'UNKNOWN'
    assert event.objects[1].track_id is None
    assert all(o.geo is None for o in event.objects)
    assert event.observed_at.isoformat().startswith('2026-11-21')


def test_ambiguous_identity_does_not_receive_id():
    a = DetectionInput('bird')
    t = SimpleNamespace(detection=a, track_id=9, frame_index=4, state='ambiguous')
    e = capture_payload(snapshot(index=4), [a], [t], 'usb', source_location=0)
    assert e.objects[0].track_id is None


def test_no_bbox_overlap_guessing():
    a = DetectionInput('bird')
    b = DetectionInput('bird')
    t = SimpleNamespace(detection=b, track_id=9, frame_index=4, state='confirmed')
    e = capture_payload(snapshot(index=4), [a], [t], 'usb', source_location=0)
    assert e.objects[0].track_id is None


def test_storage_restart_api_and_idempotency(tmp_path):
    url = f"sqlite:///{(tmp_path / 'persist.db').as_posix()}"
    db = Database(url)
    db.create_schema_for_tests_only()
    manager = FakeManager()
    writer = LiveRecorder(db, queue_capacity=8)
    writer.start(manager)
    capture(writer, frame=0)
    assert writer.drain(3)
    # Independent frames must have distinct indices; same exact frame repeats in
    # callback are NOT retried with fabricated identical host timestamps.
    capture(writer, frame=1)
    assert writer.drain(3)
    capture(writer, frame=1)  # duplicate source callback cannot double-count
    assert writer.drain(3)
    assert writer.health()['ingested_frames'] == 2
    assert writer.health()['duplicate_frames'] == 1
    assert writer.health()['errors'] == 0
    manager._state = 'stopped'
    writer.stop()
    with db.sessionmaker() as tx:
        assert tx.scalar(select(func.count()).select_from(Frame)) == 2
        assert tx.scalar(select(func.count()).select_from(Detection)) == 4
        assert tx.scalar(select(func.count()).select_from(Track)) == 1
        assert tx.scalar(select(func.count()).select_from(GeoEvidence)) == 0
        sid = tx.scalars(select(Session)).one().id
        assert tx.get(Session, sid).ended_at is not None
    db.dispose()
    reopened = Database(url)
    app = FastAPI()
    app.include_router(create_router(reopened, api_key='test-local-secret'))
    with TestClient(app) as client:
        assert client.get('/api/persistence/sessions').status_code == 401
        headers={'Authorization': 'Bearer test-local-secret'}
        assert client.get('/api/persistence/health', headers=headers).json()['status'] == 'ok'
        summary = client.get(f'/api/persistence/sessions/{sid}/summary', headers=headers).json()
        assert summary['frames'] == 2
        assert summary['detections'] == {'DRONE': 2, 'UNKNOWN': 2}
        rows = client.get(f'/api/persistence/detections?session_id={sid}', headers=headers).json()
        assert len(rows) == 4
        assert all(r['track_id'] is None for r in rows if r['classification'] == 'UNKNOWN')
        tracks=client.get(f'/api/persistence/tracks?session_id={sid}', headers=headers).json()
        assert len(tracks) == 1
        assert tracks[0]['external_track_id'] == '17'
        assert len(client.get(f"/api/persistence/tracks/{tracks[0]['id']}/history",headers=headers).json()) == 2
    reopened.dispose()


def test_rejected_malformed_source_and_duplicate_track():
    det = DetectionInput('DRONE')
    with pytest.raises(ValueError):
        capture_payload(snapshot(), [det], [], 'mystery')
    tr = SimpleNamespace(detection=det, track_id=1, frame_index=0, state='confirmed')
    tr2 = SimpleNamespace(detection=det, track_id=2, frame_index=0, state='confirmed')
    with pytest.raises(ValueError):
        capture_payload(snapshot(), [det], [tr, tr2], 'usb')


def test_no_unsafe_api_bypass(tmp_path):
    db = Database(f"sqlite:///{(tmp_path/'api.db').as_posix()}")
    db.create_schema_for_tests_only()
    app = FastAPI()
    app.include_router(create_router(db, api_key='top-secret'))
    with TestClient(app) as client:
        assert client.get('/api/persistence/cameras').status_code == 401
        assert client.get('/api/persistence/cameras', headers={'Authorization': 'Bearer wrong'}).status_code == 403
    db.dispose()
