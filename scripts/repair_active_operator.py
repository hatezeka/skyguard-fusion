"""Repair the active local backend credential, retaining its database and Vite process."""
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DB = BACKEND / ".skyguard_data" / "skyguard.sqlite3"


def listener(port):
    matches = {item.pid for item in psutil.net_connections(kind="tcp")
               if item.status == psutil.CONN_LISTEN and item.laddr.ip == "127.0.0.1"
               and item.laddr.port == port and item.pid}
    if len(matches) != 1:
        raise RuntimeError(f"Expected one local listener on port {port}")
    return psutil.Process(matches.pop())


def main():
    backend, frontend = listener(8000), listener(5173)
    assert Path(backend.cwd()) == BACKEND and Path(frontend.cwd()) == ROOT / "frontend"
    assert "app.skyguard_persistence.composed:app" in backend.cmdline()
    backend_env, frontend_env = backend.environ(), frontend.environ()
    assert not backend_env.get("SKYGUARD_DATABASE_URL") and not backend_env.get("SKYGUARD_API_KEY")
    assert any(Path(file.path) == DB for file in backend.open_files())
    media_key, db_key = frontend_env["SKYGUARD_API_KEY"], backend_env["SKYGUARD_DB_API_KEY"]
    headers = {"Authorization": "Bearer " + db_key}
    with httpx.Client(timeout=5, trust_env=False) as client:
        before = client.post("http://127.0.0.1:8000/api/media/upload")
        assert before.status_code == 503 and before.json()["detail"] == "API authentication is not configured"
        assert client.get("http://127.0.0.1:8000/api/persistence/health", headers=headers).json()["backend"] == "sqlite"
        status = client.get("http://127.0.0.1:8000/api/persistence/live/status", headers=headers)
        if status.status_code == 200 and status.json()["state"] != "idle":
            raise RuntimeError("A camera session is active")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = DB.with_name("skyguard-backup-before-auth-" + stamp + ".sqlite3")
        with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as source:
            assert source.execute("SELECT version_num FROM alembic_version").fetchone() == ("0002",)
            assert source.execute("SELECT COUNT(*) FROM sg_media_jobs WHERE status IN ('QUEUED','RUNNING')").fetchone() == (0,)
            assert source.execute("SELECT COUNT(*) FROM sg_sessions WHERE source_kind='LIVE' AND ended_at IS NULL").fetchone() == (0,)
            with closing(sqlite3.connect(backup)) as snapshot:
                source.backup(snapshot)
                assert snapshot.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                assert snapshot.execute("PRAGMA foreign_key_check").fetchall() == []
                assert snapshot.execute("SELECT version_num FROM alembic_version").fetchone() == ("0002",)

        command = [sys.executable, "-m", "uvicorn", "app.skyguard_persistence.composed:app",
                   "--host", "127.0.0.1", "--port", "8000"]

        def start(environment, label):
            log = Path(os.environ["TEMP"]) / f"skyguard-{stamp}-{label}.log"
            with log.open("wb") as output:
                return subprocess.Popen(command, cwd=BACKEND, env=environment, stdout=output, stderr=output,
                                        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)

        previous_pid = backend.pid
        assert listener(8000).pid == previous_pid
        backend.terminate()
        backend.wait(10)
        restarted = start({**backend_env, "SKYGUARD_API_KEY": media_key}, "backend")
        try:
            for _ in range(100):
                try:
                    media = client.get("http://127.0.0.1:5173/api/media/health")
                    live = client.get("http://127.0.0.1:5173/api/persistence/live/frame", headers=headers)
                    if media.status_code == 200 and live.status_code == 200:
                        break
                except httpx.RequestError:
                    pass
                if restarted.poll() is not None:
                    raise RuntimeError("Updated backend exited")
                time.sleep(.25)
            else:
                raise RuntimeError("Updated backend did not pass readiness checks")
        except BaseException:
            if restarted.poll() is None:
                restarted.terminate()
                restarted.wait(10)
            start(backend_env, "rollback")
            raise
        result = dict(previous_backend_pid=previous_pid, backend_pid=listener(8000).pid,
                      frontend_pid=frontend.pid, database=str(DB.relative_to(ROOT)),
                      database_configuration="Default SQLite path; SKYGUARD_DATABASE_URL absent",
                      backup=str(backup.relative_to(ROOT)), revision="0002",
                      prior_media_upload_http=503, media_health_through_vite_http=media.status_code,
                      configured_model_found=media.json().get("model_found"),
                      live_frame_through_vite_http=live.status_code, live_state=live.json().get("state"))
        (BACKEND / "reports" / "operator-active-runtime.json").write_text(json.dumps(result, indent=2) + "\n",
                                                                        encoding="utf-8")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
