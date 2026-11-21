"""Idempotent, explicitly synthetic SkyGuard UI fixture for the local Postgres deployment.

NEVER use as scientific evaluation evidence. No GPS, range, FPS, media or benchmark records.
Requires --seed; otherwise preflight only. No deletion or schema migration.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from sqlalchemy import inspect, select, text
from sqlalchemy.engine.url import make_url
from app.skyguard_persistence.database import Database
from app.skyguard_persistence.models import Camera, Session, Track, Detection, Frame, Alert
from app.skyguard_persistence.schemas import CameraCreate, SessionCreate, FrameIngest, ObjectInput, AlertCreate
from app.skyguard_persistence.service import Store

TAG = 'SKYGUARD_DEMO_FIXTURE_V1'


def preflight(db: Database) -> None:
    url = make_url(db.url)
    if (url.get_backend_name() != 'postgresql' or url.host != 'postgres' or
            url.database != 'skyguard' or url.username != 'skyguard'):
        raise RuntimeError('SAFETY_STOP: only internal skyguard@postgres:5432/skyguard is allowed')
    tables = set(inspect(db.engine).get_table_names())
    required = {'alembic_version','sg_cameras','sg_sessions','sg_frames','sg_detections',
                'sg_tracks','sg_track_observations','sg_alerts'}
    if not required.issubset(tables):
        raise RuntimeError('SAFETY_STOP: DB schema incomplete; never create tables implicitly')
    with db.engine.connect() as conn:
        versions = set(conn.execute(text('SELECT version_num FROM alembic_version')).scalars())
    if versions != {'0002'}:
        raise RuntimeError('SAFETY_STOP: expected Alembic revision 0002, found different schema')
    print('DEMO_DATABASE_PREFLIGHT_PASSED')


def seed(db: Database) -> str:
    store = Store(db)
    with db.sessionmaker() as tx:
        camera = tx.scalar(select(Camera).where(Camera.external_ref == TAG))
        if camera is not None and (camera.name != '[DEMO] Simulated observation camera' or
                                   camera.source_kind != 'SIMULATED'):
            raise RuntimeError('SAFETY_STOP: demo camera identity collision')
    if camera is None:
        cam = store.create_camera(CameraCreate(name='[DEMO] Simulated observation camera',
                                              source_kind='SIMULATED',external_ref=TAG,
                                              calibration_ref=None,latitude=None,longitude=None),
                                  actor='demo-fixture')
        camera_id = cam['id']
    else:
        camera_id = camera.id
    with db.sessionmaker() as tx:
        sessions = tx.scalars(select(Session).where(Session.dataset_ref == TAG)).all()
        if len(sessions) > 1:
            raise RuntimeError('SAFETY_STOP: duplicate demo sessions require investigation')
        session = sessions[0] if sessions else None
        if session is not None and (session.camera_id != camera_id or session.source_kind != 'SIMULATION' or
                                    session.validation_level != 'SOFTWARE_ONLY'):
            raise RuntimeError('SAFETY_STOP: unexpected demo session attributes')
    if session is None:
        created = store.create_session(SessionCreate(camera_id=camera_id,source_kind='SIMULATION',
                                                       validation_level='SOFTWARE_ONLY',dataset_ref=TAG),
                                       actor='demo-fixture')
        sid = created['id']
    else:
        sid = session.id
    with db.sessionmaker() as tx:
        session = tx.get(Session,sid)
        if session.ended_at is not None:
            # Re-running an already generated demo should still be harmless.
            print('DEMO_SESSION_ALREADY_CLOSED: only read-only verification')
            return sid
        existing_frames = tx.scalars(select(Frame).where(Frame.session_id == sid)).all()
        if any(f.frame_index not in range(5) for f in existing_frames):
            raise RuntimeError('SAFETY_STOP: unexpected frame added to demo session')
    base = datetime(2026,11,21,9,45,0,tzinfo=timezone.utc)
    for idx in range(5):
        items = [
            ObjectInput(classification='DRONE',confidence=0.82 + idx*0.015,
                        bbox=(135.0+idx*18,103.0+idx*8,181.0+idx*18,137.0+idx*8),
                        track_id='DEMO-DRONE-01'),
            ObjectInput(classification='BIRD',confidence=0.72 + idx*0.022,
                        bbox=(470.0-idx*11,170.0+idx*9,494.0-idx*11,189.0+idx*9),
                        track_id='DEMO-BIRD-01'),
            ObjectInput(classification='UNKNOWN',confidence=0.31,
                        bbox=(275.0+idx*2,66.0,299.0+idx*2,85.0),track_id=None)
        ]
        store.ingest_frame(FrameIngest(session_id=sid,frame_index=idx,
                                      captured_at=base+timedelta(seconds=idx), objects=items),
                           actor='demo-fixture')
    with db.sessionmaker() as tx:
        existing_alerts = tx.scalars(select(Alert).where(Alert.session_id == sid,
                                                         Alert.source == TAG)).all()
        if len(existing_alerts) not in (0,2):
            raise RuntimeError('SAFETY_STOP: incomplete existing demo alerts')
    if not existing_alerts:
        store.create_alert(AlertCreate(session_id=sid,severity='HIGH',
                          reason='[DEMO ONLY] Simulated drone classification requires operator review',
                          source=TAG),actor='demo-fixture')
        store.create_alert(AlertCreate(session_id=sid,severity='INFO',
                          reason='[DEMO ONLY] Unknown event intentionally has no track identity',
                          source=TAG),actor='demo-fixture')
    return sid


def verify(db: Database, sid: str) -> None:
    store = Store(db)
    details = store.session_summary(sid)
    dets = store.list_detections(session_id=sid,limit=100)
    tracks = store.list_tracks(session_id=sid,limit=100)
    alerts = store.list_alerts(session_id=sid,limit=100)
    if not (details['frames']==5 and len(dets)==15 and len(tracks)==2 and len(alerts)==2):
        raise RuntimeError('DEMO_VERIFICATION_FAILED: expected 5 frames, 15 detections, 2 tracks, 2 alerts')
    if {r['classification'] for r in tracks} != {'DRONE','BIRD'}:
        raise RuntimeError('DEMO_VERIFICATION_FAILED: tracks classes invalid')
    if sum(1 for d in dets if d['classification']=='UNKNOWN') != 5 or any(
            d['track_id'] is not None for d in dets if d['classification']=='UNKNOWN'):
        raise RuntimeError('DEMO_VERIFICATION_FAILED: UNKNOWN must not have tracks')
    if store.list_geo(session_id=sid,limit=100) or store.list_performance(session_id=sid,limit=100):
        raise RuntimeError('DEMO_VERIFICATION_FAILED: fixture must not contain invented geo or performance')
    print('SKYGUARD_DEMO_DB_VERIFIED')
    print('DEMO_SESSION_ID=' + sid)
    print('DEMO_FRAMES=5 DEMO_DETECTIONS=15 DEMO_TRACKS=2 DEMO_ALERTS=2')
    print('DEMO_UNKNOWN_WITHOUT_TRACK=5 DEMO_GEO=0 DEMO_PERFORMANCE=0')
    print('DEMO_ONLY_NOT_AN_AI_OR_FIELD_TEST')


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',action='store_true',help='Explicitly insert labeled synthetic local demo rows')
    args=parser.parse_args()
    if os.getenv('SKYGUARD_DEMO_SEED_ALLOWED') != 'LOCAL_ONLY_2026':
        raise RuntimeError('SAFETY_STOP: explicit local-only demo authorization missing')
    db=Database()
    try:
        preflight(db)
        if not args.seed:
            print('DRY_RUN_OK: no records changed; pass --seed to insert demo rows')
            return
        sid=seed(db)
        verify(db,sid)
    finally:
        db.dispose()

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('SKYGUARD_DEMO_ERROR: '+str(exc),file=sys.stderr)
        sys.exit(1)
