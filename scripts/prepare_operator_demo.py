"""Build an isolated SQLite demonstration database from a real recorded video."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from alembic import command
from alembic.config import Config
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
from app.detection.live_session import LiveSessionManager
from app.detection.yolo import YOLODetectionEngine
from app.skyguard_persistence.database import Database
from app.skyguard_persistence.live_capture import LiveRecorder
from app.video.source import VideoSourceConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("database", "weights", "video"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    database, weights, video = (path.resolve() for path in (args.database, args.weights, args.video))
    if not weights.is_file() or not video.is_file():
        parser.error("The real checkpoint and recorded media must exist")
    if not database.parent.is_dir() or database.suffix != ".sqlite3":
        parser.error("Use a .sqlite3 file in an existing directory")
    with database.open("xb"):
        pass
    os.environ["SKYGUARD_DATABASE_URL"] = "sqlite:///" + database.as_posix()
    os.chdir(BACKEND)
    command.upgrade(Config(str(BACKEND / "skyguard_db.ini")), "head")

    db = Database()
    manager = LiveSessionManager()
    recorder = LiveRecorder(db)
    try:
        detector = YOLODetectionEngine(model_path=str(weights), confidence_threshold=0.35)
        recorder.start(manager)
        manager.start(VideoSourceConfig("file", str(video)), detector)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if recorder.health()["ingested_frames"] >= 12 or manager.status().state in ("error", "stopped"):
                break
            time.sleep(.2)
    finally:
        try:
            manager.stop()
            recorder.stop()
        finally:
            db.dispose()

    with db.engine.begin() as connection:
        counts = {
            table: connection.execute(text("SELECT COUNT(*) FROM " + table)).scalar_one()
            for table in ("sg_frames", "sg_detections", "sg_tracks", "sg_track_observations")
        }
        if counts["sg_track_observations"] < 3 or recorder.health()["errors"]:
            raise RuntimeError("The clip did not produce at least three archived tracking observations")
        connection.execute(text("UPDATE sg_sessions SET source_kind='VIDEO', dataset_ref=:ref"),
                           {"ref": "demo:recorded:" + video.name})
        connection.execute(text("UPDATE sg_cameras SET name='Demo recording'"))
    db.dispose()
    print(json.dumps({"source": video.name, "database": str(database), "counts": counts}))


if __name__ == "__main__":
    main()
