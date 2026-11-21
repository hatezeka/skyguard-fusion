"""Local operator upload metadata backed by the existing Alembic 0002 table."""
from sqlalchemy import MetaData, Table, insert, select, update

from .database import Database
from .models import utcnow


class MediaJobs:
    def __init__(self, database: Database):
        self.database = database
        self.table = Table("sg_media_jobs", MetaData(), autoload_with=database.engine)

    def interrupt_pending(self):
        table = self.table
        with self.database.engine.begin() as connection:
            # The local worker owns only its job-relative source paths.
            connection.execute(
                update(table)
                .where(table.c.source_file.like("%/source.%"), table.c.status.in_(("RUNNING", "QUEUED")))
                .values(status="INTERRUPTED", error="Backend restarted before analysis completed", updated_at=utcnow())
            )

    def create(self, job_id, kind, source_sha256, source_file):
        now = utcnow()
        with self.database.engine.begin() as connection:
            connection.execute(insert(self.table).values(
                id=job_id, kind=kind, source_sha256=source_sha256, source_file=source_file,
                status="QUEUED", frames_processed=0, summary={}, created_at=now, updated_at=now,
            ))

    def update(self, job_id, **values):
        table = self.table
        with self.database.engine.begin() as connection:
            connection.execute(update(table).where(table.c.id == job_id).values(updated_at=utcnow(), **values))

    def get(self, job_id):
        table = self.table
        with self.database.engine.connect() as connection:
            row = connection.execute(
                select(table).where(table.c.id == job_id, table.c.source_file.like("%/source.%"))
            ).mappings().one_or_none()
        if row is None:
            return None
        return {
            "id": row["id"],
            "kind": row["kind"],
            "status": row["status"],
            "input_sha256": row["source_sha256"],
            "session_id": row["session_id"],
            "error": row["error"],
            "frames_processed": row["frames_processed"],
            "summary": row["summary"],
        }
