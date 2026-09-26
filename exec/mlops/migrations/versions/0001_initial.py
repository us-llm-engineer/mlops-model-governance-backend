"""Create model_version table.

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the model_version table."""
    op.create_table(
        "model_version",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_id", sa.String(256), nullable=False),
        sa.Column("version_id", sa.String(256), nullable=False),
        sa.Column("artifact_hash", sa.String(64), nullable=False),
        sa.Column("dataset_version", sa.String(256), nullable=True),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("validated", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("previous_production", sa.String(256), nullable=True),
        sa.Column("created_ts", sa.Float, nullable=False),
        sa.Column("updated_ts", sa.Float, nullable=False),
        sa.UniqueConstraint("model_id", "version_id", name="uq_model_version"),
        sa.CheckConstraint(
            "stage IN ('registered', 'staging', 'production', 'archived')",
            name="ck_model_version_stage"
        ),
    )


def downgrade() -> None:
    """Drop the model_version table."""
    op.drop_table("model_version")
