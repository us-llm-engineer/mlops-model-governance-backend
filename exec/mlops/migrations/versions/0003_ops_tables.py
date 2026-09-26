"""Create policy_version, incident, and drift_baseline tables.

Revision ID: 0003
Revises: 0002
Create Date: 2025-01-01 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the policy_version, incident, and drift_baseline tables."""
    # Create policy_version table
    op.create_table(
        "policy_version",
        sa.Column("version", sa.Integer, nullable=False, primary_key=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("bundle_json", sa.String, nullable=False),
        sa.Column("note", sa.String, nullable=False, server_default=""),
        sa.Column("published_ts", sa.Float, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default="0"),
    )

    # Create incident table with CHECK constraints
    op.create_table(
        "incident",
        sa.Column("incident_id", sa.String(64), nullable=False, primary_key=True),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("severity", sa.String(8), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("opened_ts", sa.Float, nullable=False),
        sa.Column("signal_json", sa.String, nullable=False),
        sa.CheckConstraint("severity IN ('P0','P1','P2','P3')", name="ck_incident_severity"),
        sa.CheckConstraint("status IN ('open','resolved')", name="ck_incident_status"),
    )

    # Create drift_baseline table with UNIQUE constraint on name
    op.create_table(
        "drift_baseline",
        sa.Column("id", sa.Integer, nullable=False, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(256), nullable=False, unique=True),
        sa.Column("reference_json", sa.String, nullable=False),
        sa.Column("created_ts", sa.Float, nullable=False),
    )


def downgrade() -> None:
    """Drop the policy_version, incident, and drift_baseline tables."""
    op.drop_table("drift_baseline")
    op.drop_table("incident")
    op.drop_table("policy_version")
