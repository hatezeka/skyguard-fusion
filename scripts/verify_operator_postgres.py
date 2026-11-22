"""Run PostgreSQL migration, schema and API checks in a disposable cluster."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))


def verify(bin_dir, temp):
    data = temp / "data"
    password = secrets.token_urlsafe(32)
    password_file = temp / "pg-password"
    password_file.write_text(password + "\n", encoding="ascii")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def pg(tool, *args):
        cmd = bin_dir / (tool + (".exe" if os.name == "nt" else ""))
        # Windows PostgreSQL children can inherit pipe handles from pg_ctl.
        # Disk-backed output allows subprocess.run to finish when pg_ctl exits.
        pg_output = temp / "pg-command.log"
        with pg_output.open("wb") as stream:
            result = subprocess.run(
                [str(cmd), *map(str, args)], stdout=stream, stderr=subprocess.STDOUT,
                timeout=90, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        if result.returncode:
            raise RuntimeError(f"{tool} failed: {pg_output.read_text(errors='replace')}")

    pg("initdb", "-D", data, "-U", "skyguard_verify", "--pwfile=" + str(password_file),
       "--auth=scram-sha-256", "--locale=C", "--encoding=UTF8")
    password_file.unlink()
    pg("pg_ctl", "-D", data, "-l", temp / "postgres.log", "-w", "start",
       "-o", f"-p {port} -h 127.0.0.1")
    engine = None
    try:
        url = f"postgresql+psycopg://skyguard_verify:{password}@127.0.0.1:{port}/postgres"
        key = secrets.token_hex(32)
        os.environ.update(SKYGUARD_DATABASE_URL=url, SKYGUARD_DB_API_KEY=key, SKYGUARD_API_KEY=key)
        os.chdir(BACKEND)
        config = Config(str(BACKEND / "skyguard_db.ini"))
        command.upgrade(config, "0001")
        engine = create_engine(url)
        with engine.begin() as conn:
            version = conn.execute(text("SHOW server_version")).scalar_one()
            conn.execute(text("INSERT INTO sg_cameras(id,name,source_kind,created_at) "
                              "VALUES ('verification-0001','Existing verification row','FILE',CURRENT_TIMESTAMP)"))
        command.upgrade(config, "0002")

        from app.skyguard_persistence.models import Base
        from app.skyguard_persistence.composed import app
        from app.skyguard_persistence.database import Database
        from app.skyguard_persistence.media_jobs import MediaJobs
        with engine.connect() as conn:
            assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "0002"
            assert conn.execute(text("SELECT name FROM sg_cameras WHERE id='verification-0001'")).scalar_one()
            metadata_diff = list(map(str, compare_metadata(
                MigrationContext.configure(conn, opts={"compare_type": True, "compare_server_default": True}),
                Base.metadata,
            )))
            inspector = inspect(conn)
            media_constraints = sorted(item["name"] for item in inspector.get_check_constraints("sg_media_jobs"))

        db = Database(url)
        try:
            jobs = MediaJobs(db)
            jobs.create("verification-job", "IMAGE", "0" * 64, "verification-job/source.jpg")
            jobs.update("verification-job", status="DONE", frames_processed=1, summary={"verification": True})
            assert jobs.get("verification-job")["summary"] == {"verification": True}
        finally:
            db.dispose()

        with TestClient(app, headers={"Authorization": f"Bearer {key}"}) as client:
            assert client.get("/api/persistence/health").json()["backend"] == "postgresql"
            assert any(row["id"] == "verification-0001" for row in client.get("/api/persistence/cameras").raise_for_status().json())
            created = client.post("/api/persistence/cameras", json={"name": "PostgreSQL API verification", "source_kind": "FILE"})
            assert created.status_code == 201, created.text
            session = client.post("/api/persistence/sessions", json={"camera_id": created.json()["id"], "source_kind": "IMAGE"})
            assert session.status_code == 201, session.text
            assert client.get("/api/persistence/live/frame").raise_for_status().json()["state"] == "idle"
            media = client.get("/api/media/jobs/verification-job", headers={"X-API-Key": key}).raise_for_status().json()
            assert media["status"] == "DONE" and media["frames_processed"] == 1

        return dict(postgresql_version=version, revision="0002", old_row_preserved=True,
                    metadata_differences=metadata_diff, media_constraints=media_constraints,
                    authenticated_api_roundtrip=True, media_json_roundtrip=True)
    finally:
        if engine is not None:
            engine.dispose()
        pg("pg_ctl", "-D", data, "-m", "fast", "-w", "stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = args.report.resolve()
    with tempfile.TemporaryDirectory(prefix="skyguard-postgres-") as temp:
        findings = verify(args.bin.resolve(), Path(temp))
    report.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    if findings["metadata_differences"]:
        raise SystemExit("PostgreSQL ORM and migration schema differ; see " + str(report))


if __name__ == "__main__":
    main()
