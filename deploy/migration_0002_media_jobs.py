"""Media input jobs: operational processing metadata, not scientific GT claims.

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('sg_media_jobs',
        sa.Column('id',sa.String(36),primary_key=True),
        sa.Column('kind',sa.String(8),nullable=False),
        sa.Column('status',sa.String(20),nullable=False),
        sa.Column('source_sha256',sa.String(64),nullable=False),
        sa.Column('source_file',sa.String(100),nullable=False),
        sa.Column('session_id',sa.String(36),sa.ForeignKey('sg_sessions.id',ondelete='SET NULL'),nullable=True),
        sa.Column('frames_processed',sa.Integer,nullable=False,server_default='0'),
        sa.Column('summary',sa.JSON,nullable=False),
        sa.Column('error',sa.String(512),nullable=True),
        sa.Column('created_at',sa.DateTime(timezone=True),nullable=False),
        sa.Column('updated_at',sa.DateTime(timezone=True),nullable=False),
        sa.CheckConstraint("kind IN ('IMAGE','VIDEO')",name='media_kind'),
        sa.CheckConstraint("status IN ('QUEUED','RUNNING','DONE','FAILED','INTERRUPTED')",name='media_status'),
        sa.CheckConstraint('frames_processed >= 0',name='media_frames_nonnegative'),
    )
    op.create_index('ix_sg_media_jobs_created_at','sg_media_jobs',['created_at'])

def downgrade():
    op.drop_index('ix_sg_media_jobs_created_at',table_name='sg_media_jobs')
    op.drop_table('sg_media_jobs')
