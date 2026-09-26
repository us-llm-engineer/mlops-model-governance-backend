"""MLOps operations repository for policy versions, incidents, and drift baselines.

Provides:
- PolicyVersionRow, IncidentRow, DriftBaselineRow ORM models (appended to mlops.sqldb.Base)
- OpsRepo for CRUD operations
"""
import json
import math
import time
from sqlalchemy import select, func
from sqlalchemy.orm import Session, mapped_column, Mapped
import sqlalchemy as sa

from mlops.kernel import ValidationFailed, NotFound, Conflict
from mlops.sqldb import Base, _check_url, _create_engine_with_pragmas, current_revision

_INT64_MAX = 2**63 - 1
# A concurrent first-insert of the same key loses the SELECT-then-INSERT race; the losing
# transaction is simply re-run (it then sees the row and updates it) instead of failing.
_MAX_WRITE_ATTEMPTS = 3

__all__ = [
    "PolicyVersionRow",
    "IncidentRow",
    "DriftBaselineRow",
    "OpsRepo",
]


class PolicyVersionRow(Base):
    """SQLAlchemy ORM model for policy_version table."""
    __tablename__ = "policy_version"

    version: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    name: Mapped[str] = mapped_column(sa.String(256), nullable=False)
    bundle_json: Mapped[str] = mapped_column(sa.String, nullable=False)
    note: Mapped[str] = mapped_column(sa.String, nullable=False, default="")
    published_ts: Mapped[float] = mapped_column(sa.Float, nullable=False)
    active: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)


class IncidentRow(Base):
    """SQLAlchemy ORM model for incident table."""
    __tablename__ = "incident"

    incident_id: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    title: Mapped[str] = mapped_column(sa.String(256), nullable=False)
    severity: Mapped[str] = mapped_column(sa.String(8), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    opened_ts: Mapped[float] = mapped_column(sa.Float, nullable=False)
    signal_json: Mapped[str] = mapped_column(sa.String, nullable=False)

    __table_args__ = (
        sa.CheckConstraint(
            "severity IN ('P0','P1','P2','P3')",
            name="ck_incident_severity"
        ),
        sa.CheckConstraint(
            "status IN ('open','resolved')",
            name="ck_incident_status"
        ),
    )


class DriftBaselineRow(Base):
    """SQLAlchemy ORM model for drift_baseline table."""
    __tablename__ = "drift_baseline"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(sa.String(256), nullable=False, unique=True)
    reference_json: Mapped[str] = mapped_column(sa.String, nullable=False)
    created_ts: Mapped[float] = mapped_column(sa.Float, nullable=False)


class OpsRepo:
    """Repository for operations persistence using SQLAlchemy."""

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
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from pathlib import Path

        cfg = Config()
        migrations_dir = Path(__file__).resolve().parent / "migrations"
        cfg.set_main_option("script_location", str(migrations_dir))

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

    def _run_write(self, work) -> None:
        """Run ``work(session)`` in one transaction with typed, value-free errors.

        IntegrityError is retried (lost insert race, see _MAX_WRITE_ATTEMPTS); the raw
        SQLAlchemy text is never propagated because it embeds the bound parameters
        (policy bundles, incident signals).
        """
        for attempt in range(_MAX_WRITE_ATTEMPTS):
            try:
                with Session(self.engine) as session, session.begin():
                    work(session)
                return
            except sa.exc.IntegrityError:
                if attempt + 1 < _MAX_WRITE_ATTEMPTS:
                    continue
                raise ValidationFailed("database constraint violation") from None
            except sa.exc.OperationalError:
                raise Conflict("database busy") from None
            except sa.exc.SQLAlchemyError:
                raise ValidationFailed("database error") from None

    def upsert_policy_version(self, record: dict) -> None:
        """Upsert a policy version record.

        Validates all fields before inserting/updating.
        Setting active=True sets every OTHER row's active to False in the SAME transaction.

        Args:
            record: dict with keys version, name, bundle_json, note, published_ts, active

        Raises:
            ValidationFailed: if any field fails validation
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        # Validate all fields
        _validate_policy_version_record(record)

        def work(session: Session) -> None:
            existing = session.scalars(
                select(PolicyVersionRow).where(PolicyVersionRow.version == record["version"])
            ).first()
            if existing:
                existing.name = record["name"]
                existing.bundle_json = record["bundle_json"]
                existing.note = record["note"]
                existing.published_ts = record["published_ts"]
                existing.active = record["active"]
            else:
                session.add(PolicyVersionRow(
                    version=record["version"],
                    name=record["name"],
                    bundle_json=record["bundle_json"],
                    note=record["note"],
                    published_ts=record["published_ts"],
                    active=record["active"],
                ))
            # Single-active invariant: deactivate every OTHER row in the SAME transaction
            # (the flush above already took SQLite's write lock, so this read is race-free).
            if record["active"]:
                for other in session.scalars(
                    select(PolicyVersionRow).where(PolicyVersionRow.version != record["version"])
                ).all():
                    other.active = False

        self._run_write(work)

    def get_policy_version(self, version: int) -> dict:
        """Get a policy version record.

        Returns a dict (copy) with all fields.

        Raises:
            NotFound: if the policy version doesn't exist
            ValidationFailed: if version is not an int in the storable range
        """
        if self._closed:
            raise RuntimeError("Repository is closed")
        _check_version(version)

        with Session(self.engine) as session:
            stmt = select(PolicyVersionRow).where(
                PolicyVersionRow.version == version
            )
            row = session.scalars(stmt).first()

            if not row:
                raise NotFound(f"policy version not found: {version}")

            return {
                "version": row.version,
                "name": row.name,
                "bundle_json": row.bundle_json,
                "note": row.note,
                "published_ts": row.published_ts,
                "active": row.active,
            }

    def list_policy_versions(self) -> list[dict]:
        """List all policy versions ordered by version ascending.

        Returns a list of dicts (copies) with all fields.
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        with Session(self.engine) as session:
            stmt = select(PolicyVersionRow).order_by(PolicyVersionRow.version)
            rows = session.scalars(stmt).all()
            result = []
            for row in rows:
                result.append({
                    "version": row.version,
                    "name": row.name,
                    "bundle_json": row.bundle_json,
                    "note": row.note,
                    "published_ts": row.published_ts,
                    "active": row.active,
                })
            return result

    def upsert_incident(self, record: dict) -> None:
        """Upsert an incident record.

        Validates all fields before inserting/updating.
        CHECK-constraint violation (bad severity/status) -> ValidationFailed.

        Args:
            record: dict with keys incident_id, title, severity, status, opened_ts, signal_json

        Raises:
            ValidationFailed: if any field fails validation
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        # Validate all fields
        _validate_incident_record(record)

        def work(session: Session) -> None:
            existing = session.scalars(
                select(IncidentRow).where(IncidentRow.incident_id == record["incident_id"])
            ).first()
            if existing:
                existing.title = record["title"]
                existing.severity = record["severity"]
                existing.status = record["status"]
                existing.opened_ts = record["opened_ts"]
                existing.signal_json = record["signal_json"]
            else:
                session.add(IncidentRow(
                    incident_id=record["incident_id"],
                    title=record["title"],
                    severity=record["severity"],
                    status=record["status"],
                    opened_ts=record["opened_ts"],
                    signal_json=record["signal_json"],
                ))

        self._run_write(work)

    def get_incident(self, incident_id: str) -> dict:
        """Get an incident record.

        Returns a dict (copy) with all fields.

        Raises:
            NotFound: if the incident doesn't exist
            ValidationFailed: if incident_id is not a text string
        """
        if self._closed:
            raise RuntimeError("Repository is closed")
        _check_text(incident_id, "incident_id")

        with Session(self.engine) as session:
            stmt = select(IncidentRow).where(
                IncidentRow.incident_id == incident_id
            )
            row = session.scalars(stmt).first()

            if not row:
                raise NotFound(f"incident not found: {incident_id}")

            return {
                "incident_id": row.incident_id,
                "title": row.title,
                "severity": row.severity,
                "status": row.status,
                "opened_ts": row.opened_ts,
                "signal_json": row.signal_json,
            }

    def count_open_incidents_by_severity(self) -> dict[str, int]:
        """Count open incidents by severity.

        Returns a dict with keys P0, P1, P2, P3, always all four keys present, 0 if none.
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        with Session(self.engine) as session:
            # Query: SELECT severity, COUNT(*) FROM incident WHERE status='open' GROUP BY severity
            stmt = select(
                IncidentRow.severity,
                func.count().label("count")
            ).where(
                IncidentRow.status == 'open'
            ).group_by(IncidentRow.severity)

            results = session.execute(stmt).all()

            # Build dict with all severities, default to 0
            counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
            for severity, count in results:
                counts[severity] = count

            return counts

    def upsert_drift_baseline(self, name: str, reference: list[float]) -> None:
        """Upsert a drift baseline record.

        name must be non-empty str<=256 no control chars.
        reference must be >=30 finite floats else ValidationFailed.
        UNIQUE-violation on name -> update in place (upsert, not reject).

        Args:
            name: baseline name (str)
            reference: list of floats for the reference

        Raises:
            ValidationFailed: if validation fails
        """
        if self._closed:
            raise RuntimeError("Repository is closed")

        # Validate inputs
        _validate_drift_baseline_inputs(name, reference)

        reference_json = json.dumps(reference)

        def work(session: Session) -> None:
            existing = session.scalars(
                select(DriftBaselineRow).where(DriftBaselineRow.name == name)
            ).first()
            if existing:
                existing.reference_json = reference_json  # created_ts is kept
            else:
                session.add(DriftBaselineRow(
                    name=name, reference_json=reference_json, created_ts=time.time(),
                ))

        self._run_write(work)

    def get_drift_baseline(self, name: str) -> list[float]:
        """Get a drift baseline record.

        Returns list of floats.

        Raises:
            NotFound: if the drift baseline doesn't exist
            ValidationFailed: if name is not a text string
        """
        if self._closed:
            raise RuntimeError("Repository is closed")
        _check_text(name, "name")

        with Session(self.engine) as session:
            stmt = select(DriftBaselineRow).where(
                DriftBaselineRow.name == name
            )
            row = session.scalars(stmt).first()

            if not row:
                raise NotFound(f"drift baseline not found: {name}")

            return json.loads(row.reference_json)


def _validate_policy_version_record(record: dict) -> None:
    """Validate a policy version record.

    Raises ValidationFailed if any field is invalid.
    """
    required_fields = {
        "version", "name", "bundle_json", "note", "published_ts", "active"
    }

    if not isinstance(record, dict):
        raise ValidationFailed("record must be a dict")

    missing = required_fields - set(record.keys())
    if missing:
        raise ValidationFailed(f"missing fields: {missing}")

    unknown = set(record.keys()) - required_fields
    if unknown:
        raise ValidationFailed(f"unknown fields: {unknown}")

    # Validate version (int, not bool, 1..int64)
    _check_version(record["version"])
    if record["version"] < 1:
        raise ValidationFailed("version must be >= 1")

    # Validate name (non-empty string, <=256, no control chars)
    _check_text(record["name"], "name", clean=True)
    if not record["name"]:
        raise ValidationFailed("name must be a non-empty string")
    if len(record["name"]) > 256:
        raise ValidationFailed("name must be <= 256 characters")

    # Validate bundle_json (non-empty string)
    _check_text(record["bundle_json"], "bundle_json")
    if not record["bundle_json"]:
        raise ValidationFailed("bundle_json must be a non-empty string")

    # Validate note (string)
    _check_text(record["note"], "note")

    # Validate published_ts (finite number, not bool)
    _check_finite_number(record["published_ts"], "published_ts")

    # Validate active (bool, not int)
    if not isinstance(record["active"], bool):
        raise ValidationFailed("active must be a bool")


def _validate_incident_record(record: dict) -> None:
    """Validate an incident record.

    Raises ValidationFailed if any field is invalid.
    """
    required_fields = {
        "incident_id", "title", "severity", "status", "opened_ts", "signal_json"
    }

    if not isinstance(record, dict):
        raise ValidationFailed("record must be a dict")

    missing = required_fields - set(record.keys())
    if missing:
        raise ValidationFailed(f"missing fields: {missing}")

    unknown = set(record.keys()) - required_fields
    if unknown:
        raise ValidationFailed(f"unknown fields: {unknown}")

    # Validate incident_id (non-empty string, <=64, no control chars)
    _check_text(record["incident_id"], "incident_id", clean=True)
    if not record["incident_id"]:
        raise ValidationFailed("incident_id must be a non-empty string")
    if len(record["incident_id"]) > 64:
        raise ValidationFailed("incident_id must be <= 64 characters")

    # Validate title (non-empty string, <=256, no control chars)
    _check_text(record["title"], "title", clean=True)
    if not record["title"]:
        raise ValidationFailed("title must be a non-empty string")
    if len(record["title"]) > 256:
        raise ValidationFailed("title must be <= 256 characters")

    # Validate severity / status (str first: an unhashable value must not reach `in <set>`)
    if not isinstance(record["severity"], str) or record["severity"] not in {"P0", "P1", "P2", "P3"}:
        raise ValidationFailed("severity must be one of P0, P1, P2, P3")
    if not isinstance(record["status"], str) or record["status"] not in {"open", "resolved"}:
        raise ValidationFailed("status must be one of open, resolved")

    # Validate opened_ts (finite number, not bool)
    _check_finite_number(record["opened_ts"], "opened_ts")

    # Validate signal_json (non-empty string)
    _check_text(record["signal_json"], "signal_json")
    if not record["signal_json"]:
        raise ValidationFailed("signal_json must be a non-empty string")


def _validate_drift_baseline_inputs(name: str, reference: list) -> None:
    """Validate drift baseline inputs.

    Raises ValidationFailed if any input is invalid.
    """
    # Validate name (non-empty string, <=256, no control chars)
    _check_text(name, "name")
    if not name:
        raise ValidationFailed("name must be a non-empty string")
    if len(name) > 256:
        raise ValidationFailed("name must be <= 256 characters")
    if not _is_clean_string(name):
        raise ValidationFailed("name contains control characters")

    # Validate reference (>=30 finite floats)
    if not isinstance(reference, list):
        raise ValidationFailed("reference must be a list")
    if len(reference) < 30:
        raise ValidationFailed("reference must have at least 30 elements")

    for i, val in enumerate(reference):
        _check_finite_number(val, f"reference[{i}]")


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


def _check_version(version) -> None:
    """int (not bool) within SQLite's INTEGER range, else ValidationFailed."""
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValidationFailed("version must be an int, not bool")
    if abs(version) > _INT64_MAX:
        raise ValidationFailed("version out of range")


def _check_text(value, field: str, *, clean: bool = False) -> None:
    """str that UTF-8-encodes (no lone surrogates); ``clean`` also forbids control chars."""
    if not isinstance(value, str):
        raise ValidationFailed(f"{field} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValidationFailed(f"{field} is not valid text") from None
    if clean and not _is_clean_string(value):
        raise ValidationFailed(f"{field} contains control characters")


def _check_finite_number(value, field: str) -> None:
    """Finite int/float that is representable as a float (huge ints overflow), not bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationFailed(f"{field} must be a numeric value")
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        raise ValidationFailed(f"{field} must be finite")
