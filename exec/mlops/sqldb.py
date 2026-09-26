"""SQLAlchemy models and repository for MLOps database.

Provides:
- ModelVersionRow and DatasetVersionRow ORM models
- SqlRepo for CRUD operations
- Alembic migration functions (upgrade, downgrade, current_revision)
- SQLite pragmas (WAL, foreign_keys, busy_timeout)
"""
import re
import time
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from alembic.config import Config
from alembic import command
from alembic.util.exc import CommandError

from mlops.kernel import ValidationFailed, NotFound, canonical_json

__all__ = [
    "ModelVersionRow",
    "DatasetVersionRow",
    "SqlRepo",
    "upgrade",
    "downgrade",
    "current_revision",
]

# SQLAlchemy ORM base
class Base(DeclarativeBase):
    pass


class ModelVersionRow(Base):
    """SQLAlchemy ORM model for model_version table."""
    __tablename__ = "model_version"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(sa.String(256), nullable=False)
    version_id: Mapped[str] = mapped_column(sa.String(256), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    dataset_version: Mapped[str | None] = mapped_column(sa.String(256), nullable=True)
    stage: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    validated: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    previous_production: Mapped[str | None] = mapped_column(sa.String(256), nullable=True)
    created_ts: Mapped[float] = mapped_column(sa.Float, nullable=False)
    updated_ts: Mapped[float] = mapped_column(sa.Float, nullable=False)

    __table_args__ = (
        sa.UniqueConstraint("model_id", "version_id", name="uq_model_version"),
        sa.CheckConstraint(
            "stage IN ('registered', 'staging', 'production', 'archived')",
            name="ck_model_version_stage"
        ),
    )


class DatasetVersionRow(Base):
    """SQLAlchemy ORM model for dataset_version table."""
    __tablename__ = "dataset_version"

    version_id: Mapped[str] = mapped_column(sa.String(256), primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(256), nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    rows: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    schema_json: Mapped[str] = mapped_column(sa.String, nullable=False)


def _create_engine_with_pragmas(url: str):
    """Create a SQLAlchemy engine with SQLite pragmas configured."""
    engine = sa.create_engine(url)

    # Set up SQLite pragmas
    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, connection_record):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA foreign_keys=ON")
        dbapi_conn.execute("PRAGMA busy_timeout=5000")

    return engine


class SqlRepo:
    """Repository for model and dataset persistence using SQLAlchemy."""

    def __init__(self, url: str) -> None:
        """Initialize the repository and verify database is migrated.

        Raises ValidationFailed if database is not at head migration.
        """
        _check_url(url)

        self.engine = _create_engine_with_pragmas(url)
        self._closed = False

        # Check if database is migrated to head
        rev = current_revision(url)
        if rev is None:
            raise ValidationFailed("database not migrated")

        # Get head revision
        cfg = Config()
        migrations_dir = Path(__file__).resolve().parent / "migrations"
        cfg.set_main_option("script_location", str(migrations_dir))

        from alembic.script import ScriptDirectory
        script = ScriptDirectory.from_config(cfg)

        # Get the head revision
        head_revisions = script.get_heads()
        if not head_revisions or rev not in head_revisions:
            raise ValidationFailed("database not migrated")

    def close(self) -> None:
        """Close the database connection pool."""
        if not self._closed:
            self.engine.dispose()
            self._closed = True

    def upsert_model(self, record: dict) -> None:
        """Upsert a model version record.

        Validates all fields before inserting/updating.
        On update of existing (model_id, version_id):
        - Preserves created_ts (immutable)
        - Enforces artifact_hash immutability (raises ValidationFailed if different)
        - Keeps updated_ts at max(old, new) to never go backwards
        - Allows changes to: stage, validated, previous_production, dataset_version

        Args:
            record: dict with keys model_id, version_id, artifact_hash, dataset_version,
                   stage, validated, previous_production, created_ts, updated_ts

        Raises:
            ValidationFailed: if any field fails validation, artifact_hash differs on update,
            (translates sqlalchemy.exc.IntegrityError to ValidationFailed for CHECK/UNIQUE)
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        # Validate all fields
        _validate_model_record(record)

        try:
            with Session(self.engine) as session, session.begin():
                # Check if record exists
                stmt = select(ModelVersionRow).where(
                    (ModelVersionRow.model_id == record["model_id"]) &
                    (ModelVersionRow.version_id == record["version_id"])
                )
                existing = session.scalars(stmt).first()

                if existing:
                    # Validate artifact_hash immutability
                    if existing.artifact_hash != record["artifact_hash"]:
                        raise ValidationFailed("artifact_hash is immutable")

                    # Update existing record with explicit field assignments
                    # created_ts is preserved (immutable)
                    # updated_ts never goes backwards (keep max of old and new)
                    existing.dataset_version = record["dataset_version"]
                    existing.stage = record["stage"]
                    existing.validated = record["validated"]
                    existing.previous_production = record["previous_production"]
                    existing.updated_ts = max(existing.updated_ts, record["updated_ts"])
                else:
                    # Insert new record
                    row = ModelVersionRow(
                        model_id=record["model_id"],
                        version_id=record["version_id"],
                        artifact_hash=record["artifact_hash"],
                        dataset_version=record["dataset_version"],
                        stage=record["stage"],
                        validated=record["validated"],
                        previous_production=record["previous_production"],
                        created_ts=record["created_ts"],
                        updated_ts=record["updated_ts"],
                    )
                    session.add(row)
        except sa.exc.IntegrityError as e:
            # Translate database integrity errors to ValidationFailed
            raise ValidationFailed(f"database constraint violation: {str(e)}")

    def get_model(self, model_id: str) -> list[dict]:
        """Get all versions of a model, ordered by created_ts.

        Returns a list of dicts (copies) with all fields.
        Returns empty list if model not found.
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        # Validate model_id
        if not isinstance(model_id, str) or not model_id:
            raise ValidationFailed("model_id must be a non-empty string")
        if len(model_id) > 256:
            raise ValidationFailed("model_id must be <= 256 characters")
        if not _is_clean_string(model_id):
            raise ValidationFailed("model_id contains control characters")

        with Session(self.engine) as session:
            stmt = select(ModelVersionRow).where(
                ModelVersionRow.model_id == model_id
            ).order_by(ModelVersionRow.created_ts)

            rows = session.scalars(stmt).all()
            result = []
            for row in rows:
                result.append({
                    "model_id": row.model_id,
                    "version_id": row.version_id,
                    "artifact_hash": row.artifact_hash,
                    "dataset_version": row.dataset_version,
                    "stage": row.stage,
                    "validated": row.validated,
                    "previous_production": row.previous_production,
                    "created_ts": row.created_ts,
                    "updated_ts": row.updated_ts,
                })
            return result

    def set_stage(self, model_id: str, version_id: str, stage: str,
                  previous_production: str | None = None) -> None:
        """Set the stage of a model version.

        Updates updated_ts to current time.

        Raises:
            NotFound: if the model version doesn't exist
            ValidationFailed: if stage is invalid
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        # Validate model_id and version_id
        if not isinstance(model_id, str) or not model_id:
            raise ValidationFailed("model_id must be a non-empty string")
        if len(model_id) > 256:
            raise ValidationFailed("model_id must be <= 256 characters")
        if not _is_clean_string(model_id):
            raise ValidationFailed("model_id contains control characters")

        if not isinstance(version_id, str) or not version_id:
            raise ValidationFailed("version_id must be a non-empty string")
        if len(version_id) > 256:
            raise ValidationFailed("version_id must be <= 256 characters")
        if not _is_clean_string(version_id):
            raise ValidationFailed("version_id contains control characters")

        # Validate stage
        valid_stages = {"registered", "staging", "production", "archived"}
        if stage not in valid_stages:
            raise ValidationFailed(f"invalid stage: {stage}")

        try:
            with Session(self.engine) as session, session.begin():
                stmt = select(ModelVersionRow).where(
                    (ModelVersionRow.model_id == model_id) &
                    (ModelVersionRow.version_id == version_id)
                )
                row = session.scalars(stmt).first()

                if not row:
                    raise NotFound(f"model version not found: {model_id}/{version_id}")

                row.stage = stage
                row.previous_production = previous_production
                row.updated_ts = time.time()
        except sa.exc.IntegrityError as e:
            raise ValidationFailed(f"database constraint violation: {str(e)}")

    def upsert_dataset(self, record: dict) -> None:
        """Upsert a dataset version record.

        Validates all fields before inserting/updating.
        schema_json is stored as canonical JSON.

        Args:
            record: dict with keys version_id, name, content_hash, rows, schema_json

        Raises:
            ValidationFailed: if any field fails validation
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        # Validate all fields
        _validate_dataset_record(record)

        try:
            with Session(self.engine) as session, session.begin():
                # Check if record exists
                stmt = select(DatasetVersionRow).where(
                    DatasetVersionRow.version_id == record["version_id"]
                )
                existing = session.scalars(stmt).first()

                # Store schema_json as canonical JSON
                schema_json_str = canonical_json(record["schema_json"])

                if existing:
                    # Update existing record
                    existing.name = record["name"]
                    existing.content_hash = record["content_hash"]
                    existing.rows = record["rows"]
                    existing.schema_json = schema_json_str
                else:
                    # Insert new record
                    row = DatasetVersionRow(
                        version_id=record["version_id"],
                        name=record["name"],
                        content_hash=record["content_hash"],
                        rows=record["rows"],
                        schema_json=schema_json_str,
                    )
                    session.add(row)
        except sa.exc.IntegrityError as e:
            raise ValidationFailed(f"database constraint violation: {str(e)}")

    def get_dataset(self, version_id: str) -> dict:
        """Get a dataset version record.

        Returns a dict (copy) with schema_json deserialized.

        Raises:
            NotFound: if the dataset version doesn't exist
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        # Validate version_id
        if not isinstance(version_id, str) or not version_id:
            raise ValidationFailed("version_id must be a non-empty string")
        if len(version_id) > 256:
            raise ValidationFailed("version_id must be <= 256 characters")
        if not _is_clean_string(version_id):
            raise ValidationFailed("version_id contains control characters")

        import json

        with Session(self.engine) as session:
            stmt = select(DatasetVersionRow).where(
                DatasetVersionRow.version_id == version_id
            )
            row = session.scalars(stmt).first()

            if not row:
                raise NotFound(f"dataset version not found: {version_id}")

            return {
                "version_id": row.version_id,
                "name": row.name,
                "content_hash": row.content_hash,
                "rows": row.rows,
                "schema_json": json.loads(row.schema_json),
            }

    def count(self) -> int:
        """Count the total number of model versions."""
        if self._closed:
            raise RuntimeError("Repository is closed")

        with Session(self.engine) as session:
            stmt = select(sa.func.count()).select_from(ModelVersionRow)
            return session.scalars(stmt).first() or 0


def _validate_model_record(record: dict) -> None:
    """Validate a model record.

    Raises ValidationFailed if any field is invalid.
    """
    required_fields = {
        "model_id", "version_id", "artifact_hash", "dataset_version",
        "stage", "validated", "previous_production", "created_ts", "updated_ts"
    }

    if not isinstance(record, dict):
        raise ValidationFailed("record must be a dict")

    missing = required_fields - set(record.keys())
    if missing:
        raise ValidationFailed(f"missing fields: {missing}")

    unknown = set(record.keys()) - required_fields
    if unknown:
        raise ValidationFailed(f"unknown fields: {unknown}")

    # Validate model_id (non-empty string, <=256, no control chars)
    if not isinstance(record["model_id"], str) or not record["model_id"]:
        raise ValidationFailed("model_id must be a non-empty string")
    if len(record["model_id"]) > 256:
        raise ValidationFailed("model_id must be <= 256 characters")
    if not _is_clean_string(record["model_id"]):
        raise ValidationFailed("model_id contains control characters")

    # Validate version_id (non-empty string, <=256, no control chars)
    if not isinstance(record["version_id"], str) or not record["version_id"]:
        raise ValidationFailed("version_id must be a non-empty string")
    if len(record["version_id"]) > 256:
        raise ValidationFailed("version_id must be <= 256 characters")
    if not _is_clean_string(record["version_id"]):
        raise ValidationFailed("version_id contains control characters")

    # Validate artifact_hash (64 lowercase hex chars)
    if not isinstance(record["artifact_hash"], str):
        raise ValidationFailed("artifact_hash must be a string")
    if not re.fullmatch(r"[a-f0-9]{64}", record["artifact_hash"]):
        raise ValidationFailed("artifact_hash must be 64 lowercase hex characters")

    # Validate dataset_version (optional, None or string)
    if record["dataset_version"] is not None:
        if not isinstance(record["dataset_version"], str):
            raise ValidationFailed("dataset_version must be None or a string")

    # Validate stage (one of the 4 values)
    valid_stages = {"registered", "staging", "production", "archived"}
    if record["stage"] not in valid_stages:
        raise ValidationFailed(f"stage must be one of {valid_stages}")

    # Validate validated (bool, not int/float)
    if not isinstance(record["validated"], bool):
        raise ValidationFailed("validated must be a bool")

    # Validate previous_production (optional, None or string)
    if record["previous_production"] is not None:
        if not isinstance(record["previous_production"], str):
            raise ValidationFailed("previous_production must be None or a string")

    # Validate created_ts (float, not NaN/inf/bool/str)
    if not isinstance(record["created_ts"], (int, float)) or isinstance(record["created_ts"], bool):
        raise ValidationFailed("created_ts must be a numeric value")
    if isinstance(record["created_ts"], float):
        if record["created_ts"] != record["created_ts"]:  # NaN check
            raise ValidationFailed("created_ts cannot be NaN")
        if record["created_ts"] == float('inf') or record["created_ts"] == float('-inf'):
            raise ValidationFailed("created_ts cannot be infinite")

    # Validate updated_ts (float, not NaN/inf/bool/str)
    if not isinstance(record["updated_ts"], (int, float)) or isinstance(record["updated_ts"], bool):
        raise ValidationFailed("updated_ts must be a numeric value")
    if isinstance(record["updated_ts"], float):
        if record["updated_ts"] != record["updated_ts"]:  # NaN check
            raise ValidationFailed("updated_ts cannot be NaN")
        if record["updated_ts"] == float('inf') or record["updated_ts"] == float('-inf'):
            raise ValidationFailed("updated_ts cannot be infinite")


def _validate_dataset_record(record: dict) -> None:
    """Validate a dataset record.

    Raises ValidationFailed if any field is invalid.
    """
    required_fields = {"version_id", "name", "content_hash", "rows", "schema_json"}

    if not isinstance(record, dict):
        raise ValidationFailed("record must be a dict")

    missing = required_fields - set(record.keys())
    if missing:
        raise ValidationFailed(f"missing fields: {missing}")

    unknown = set(record.keys()) - required_fields
    if unknown:
        raise ValidationFailed(f"unknown fields: {unknown}")

    # Validate version_id (non-empty string, <=256)
    if not isinstance(record["version_id"], str) or not record["version_id"]:
        raise ValidationFailed("version_id must be a non-empty string")
    if len(record["version_id"]) > 256:
        raise ValidationFailed("version_id must be <= 256 characters")

    # Validate name (non-empty string)
    if not isinstance(record["name"], str) or not record["name"]:
        raise ValidationFailed("name must be a non-empty string")

    # Validate content_hash (64 lowercase hex chars)
    if not isinstance(record["content_hash"], str):
        raise ValidationFailed("content_hash must be a string")
    if not re.fullmatch(r"[a-f0-9]{64}", record["content_hash"]):
        raise ValidationFailed("content_hash must be 64 lowercase hex characters")

    # Validate rows (non-negative integer)
    if not isinstance(record["rows"], int) or isinstance(record["rows"], bool):
        raise ValidationFailed("rows must be an integer")
    if record["rows"] < 0:
        raise ValidationFailed("rows must be non-negative")

    # Validate schema_json (dict with string keys and valid values)
    if not isinstance(record["schema_json"], dict):
        raise ValidationFailed("schema_json must be a dict")

    # Try to serialize to canonical JSON to validate it
    try:
        canonical_json(record["schema_json"])
    except (ValueError, TypeError) as e:
        raise ValidationFailed(f"schema_json is not valid: {e}")


def _check_url(url: str) -> None:
    """Validate database URL.

    Raises ValidationFailed if URL is invalid or not SQLite.
    """
    if not isinstance(url, str) or not url:
        raise ValidationFailed("database URL must be a non-empty string")
    if len(url) > 2048:
        raise ValidationFailed("database URL is too long")
    if not _is_clean_string(url):
        raise ValidationFailed("database URL contains control characters")
    if not url.startswith("sqlite:///"):
        raise ValidationFailed("only SQLite databases are supported")


def _check_revision(revision: str) -> None:
    """Validate migration revision ID.

    Raises ValidationFailed if revision is invalid.
    """
    if not isinstance(revision, str):
        raise ValidationFailed("revision must be a string")
    if not re.match(r"^[A-Za-z0-9_@+.\-]{1,64}$", revision):
        raise ValidationFailed("invalid revision format")


def _is_clean_string(s: str) -> bool:
    """Check if a string contains no control characters."""
    if not isinstance(s, str):
        return False
    # Check for control characters (0x00-0x1f and 0x7f-0x9f)
    for char in s:
        code = ord(char)
        if code < 0x20 or (0x7f <= code <= 0x9f):
            return False
    return True


# Alembic migration functions
def upgrade(url: str, revision: str = "head") -> str | None:
    """Run migrations up to the specified revision.

    Args:
        url: database URL
        revision: target revision (default "head")

    Returns:
        The head revision id string, or None if no migrations
    """
    _check_url(url)
    _check_revision(revision)

    try:
        # Create engine with pragmas so the database is created with WAL mode
        engine = _create_engine_with_pragmas(url)

        cfg = Config()
        migrations_dir = Path(__file__).resolve().parent / "migrations"
        cfg.set_main_option("script_location", str(migrations_dir))
        cfg.set_main_option("sqlalchemy.url", url)

        # Create a connection to ensure pragmas are applied before migrations
        with engine.begin():
            pass

        command.upgrade(cfg, revision)
        engine.dispose()

        # Return the head revision
        from alembic.script import ScriptDirectory
        script = ScriptDirectory.from_config(cfg)
        heads = script.get_heads()
        return heads[0] if heads else None
    except CommandError as e:
        raise ValidationFailed("migration error")
    except (sa.exc.ArgumentError, sa.exc.OperationalError) as e:
        raise ValidationFailed("database error")


def downgrade(url: str, revision: str = "base") -> str | None:
    """Run migrations down to the specified revision.

    Args:
        url: database URL
        revision: target revision (default "base")

    Returns:
        The current revision id after downgrade, or None at base
    """
    _check_url(url)
    _check_revision(revision)

    try:
        cfg = Config()
        migrations_dir = Path(__file__).resolve().parent / "migrations"
        cfg.set_main_option("script_location", str(migrations_dir))
        cfg.set_main_option("sqlalchemy.url", url)

        command.downgrade(cfg, revision)

        return current_revision(url)
    except CommandError as e:
        raise ValidationFailed("migration error")
    except (sa.exc.ArgumentError, sa.exc.OperationalError) as e:
        raise ValidationFailed("database error")


def current_revision(url: str) -> str | None:
    """Get the current migration revision of a database.

    Args:
        url: database URL

    Returns:
        The current revision id, or None if no migrations have been run
    """
    _check_url(url)

    try:
        # Create a temporary engine to check the alembic_version table
        engine = sa.create_engine(url)
        try:
            with engine.connect() as conn:
                try:
                    result = conn.exec_driver_sql(
                        "SELECT version_num FROM alembic_version ORDER BY version_num DESC LIMIT 1"
                    ).fetchone()
                    return result[0] if result else None
                except sa.exc.OperationalError:
                    # Table doesn't exist yet
                    return None
        finally:
            engine.dispose()
    except sa.exc.ArgumentError as e:
        raise ValidationFailed("database error")
