"""Database engine + sessions. Schema upgrades are executed via Alembic CLI."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .models import Base


def default_database_url() -> str:
    root = Path(__file__).resolve().parents[2]  # backend/
    path = root / ".skyguard_data" / "skyguard.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


class Database:
    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("SKYGUARD_DATABASE_URL") or default_database_url()
        params: dict = {"pool_pre_ping": True}
        if self.url.startswith("sqlite"):
            params["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine: Engine = create_engine(self.url, **params)
        if self.engine.dialect.name == "sqlite":
            @event.listens_for(self.engine, "connect")
            def sqlite_pragmas(connection, _connection_record):
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
        self.sessionmaker = sessionmaker(self.engine, autoflush=False, expire_on_commit=False)

    def create_schema_for_tests_only(self) -> None:
        """Testing-only bootstrap. Use Alembic revisions for real deployments."""
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()
