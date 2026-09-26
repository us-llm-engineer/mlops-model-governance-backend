"""Create dataset_version table.

Revision ID: 0002
Revises: 0001
Create Date: 2025-01-01 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the dataset_version table."""
    op.create_table(
        "dataset_version",
        sa.Column("version_id", sa.String(256), nullable=False, primary_key=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("rows", sa.Integer, nullable=False),
        sa.Column("schema_json", sa.String, nullable=False),
    )


def downgrade() -> None:
    """Drop the dataset_version table."""
    op.drop_table("dataset_version")
