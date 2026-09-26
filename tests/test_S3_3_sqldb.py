"""S3 wave 2 - mlops.sqldb contract (SqlRepo, Alembic migrations). Written BEFORE the code."""
import math
import sqlite3
import threading
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.script import ScriptDirectory

from mlops.kernel import NotFound, ValidationFailed

import mlops.sqldb as sqldb  # noqa: E402

MIG_DIR = Path(sqldb.__file__).resolve().parent / "migrations"
HASH = "a" * 64


def _url(tmp_path, name="t.db"):
    return f"sqlite:///{tmp_path}/{name}"


def _tables(url):
    eng = sa.create_engine(url)
    try:
        return set(sa.inspect(eng).get_table_names())
    finally:
        eng.dispose()


def _revisions():
    """Ordered revision ids (base -> head) read from the migrations dir."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIG_DIR))
    script = ScriptDirectory.from_config(cfg)
    revs = list(script.walk_revisions("base", "heads"))
    return [r.revision for r in reversed(revs)], script


def _rec(model_id="m", version_id="v1", created=1.0, **kw):
    r = {"model_id": model_id, "version_id": version_id, "artifact_hash": HASH,
         "dataset_version": None, "stage": "registered", "validated": False,
         "previous_production": None, "created_ts": created, "updated_ts": created}
    r.update(kw)
    return r


def _ds(version_id="d1", **kw):
    r = {"version_id": version_id, "name": "clicks", "content_hash": HASH,
         "rows": 10, "schema_json": {"b": "int", "a": "str"}}
    r.update(kw)
    return r


@pytest.fixture
def migrated(tmp_path):
    url = _url(tmp_path)
    sqldb.upgrade(url)
    return url


@pytest.fixture
def repo(migrated):
    r = sqldb.SqlRepo(migrated)
    yield r
    r.close()


# ---------------------------------------------------------------- migrations
def test_migration_files_exist_and_chain():
    assert (MIG_DIR / "env.py").is_file()
    assert (MIG_DIR / "script.py.mako").is_file()
    files = [p for p in (MIG_DIR / "versions").glob("*.py") if p.name != "__init__.py"]
    assert len(files) >= 2
    revs, script = _revisions()
    assert len(revs) >= 2
    # linear chain: each non-first revision's down_revision is the previous one
    for prev, cur in zip(revs, revs[1:]):
        assert script.get_revision(cur).down_revision == prev
    assert script.get_revision(revs[0]).down_revision is None


def test_fresh_db_has_no_revision_and_upgrade_head_idempotent(tmp_path):
    url = _url(tmp_path)
    assert sqldb.current_revision(url) is None
    revs, _ = _revisions()
    head = sqldb.upgrade(url)
    assert isinstance(sqldb.current_revision(url), str) and sqldb.current_revision(url)
    assert sqldb.current_revision(url) == revs[-1]
    again = sqldb.upgrade(url)
    assert sqldb.current_revision(url) == revs[-1]
    assert head == again  # documented: upgrade returns None or the head; both calls agree


def test_tables_created_at_head(migrated):
    names = _tables(migrated)
    assert {sqldb.ModelVersionRow.__tablename__, sqldb.DatasetVersionRow.__tablename__,
            "alembic_version"} <= names


def test_partial_upgrade_and_downgrade_steps(tmp_path):
    revs, _ = _revisions()
    url = _url(tmp_path)
    sqldb.upgrade(url, revs[0])
    assert sqldb.current_revision(url) == revs[0]
    names = _tables(url)
    assert sqldb.ModelVersionRow.__tablename__ in names
    assert sqldb.DatasetVersionRow.__tablename__ not in names  # 0002 adds datasets
    sqldb.upgrade(url)
    assert sqldb.DatasetVersionRow.__tablename__ in _tables(url)
    sqldb.downgrade(url, revs[0])
    assert sqldb.current_revision(url) == revs[0]
    assert sqldb.DatasetVersionRow.__tablename__ not in _tables(url)


def test_downgrade_base_removes_tables(migrated):
    sqldb.downgrade(migrated, "base")
    assert sqldb.current_revision(migrated) is None
    names = _tables(migrated)
    assert sqldb.ModelVersionRow.__tablename__ not in names
    assert sqldb.DatasetVersionRow.__tablename__ not in names


# ------------------------------------------------------- unmigrated refusal
def test_unmigrated_db_refused_and_no_tables_created(tmp_path):
    url = _url(tmp_path)
    with pytest.raises(ValidationFailed, match="database not migrated"):
        sqldb.SqlRepo(url)
    assert _tables(url) == set()


def test_old_revision_refused(tmp_path):
    revs, _ = _revisions()
    url = _url(tmp_path)
    sqldb.upgrade(url, revs[0])
    with pytest.raises(ValidationFailed, match="database not migrated"):
        sqldb.SqlRepo(url)
    assert sqldb.DatasetVersionRow.__tablename__ not in _tables(url)


# ------------------------------------------------------------------- pragmas
def test_pragmas(repo, tmp_path):
    with sqlite3.connect(f"{tmp_path}/t.db") as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    eng = getattr(repo, "engine", None)
    assert eng is not None, "SqlRepo must expose .engine (used to verify pragmas)"
    with eng.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar() >= 5000
        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"


# ---------------------------------------------------------------------- CRUD
def test_model_upsert_get_ordering_and_copies(repo):
    repo.upsert_model(_rec(version_id="v2", created=2.0))
    repo.upsert_model(_rec(version_id="v1", created=1.0))
    got = repo.get_model("m")
    assert [r["version_id"] for r in got] == ["v1", "v2"]
    assert got[0]["stage"] == "registered" and got[0]["validated"] is False
    assert got[0]["previous_production"] is None and got[0]["dataset_version"] is None
    got[0]["stage"] = "production"
    got.append({"x": 1})
    again = repo.get_model("m")
    assert len(again) == 2 and again[0]["stage"] == "registered"
    assert repo.get_model("unknown") == []


def test_upsert_same_key_updates_in_place(repo):
    repo.upsert_model(_rec())
    assert repo.count() == 1
    repo.upsert_model(_rec(validated=True, updated_ts=5.0))
    assert repo.count() == 1
    row = repo.get_model("m")[0]
    assert row["validated"] is True and row["updated_ts"] == 5.0


def test_set_stage(repo):
    repo.upsert_model(_rec(created=1.0))
    repo.set_stage("m", "v1", "production", previous_production="v0")
    row = repo.get_model("m")[0]
    assert row["stage"] == "production" and row["previous_production"] == "v0"
    assert row["updated_ts"] > 1.0
    with pytest.raises(NotFound):
        repo.set_stage("m", "nope", "staging")
    with pytest.raises(NotFound):
        repo.set_stage("nomodel", "v1", "staging")


def test_bad_stage_is_validation_failed_not_raw_sqlalchemy(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_model(_rec(stage="bogus"))
    assert repo.count() == 0
    repo.upsert_model(_rec())
    with pytest.raises(ValidationFailed):
        repo.set_stage("m", "v1", "bogus")
    assert repo.get_model("m")[0]["stage"] == "registered"


def test_check_constraint_enforced_by_database(repo):
    """Belt and braces: even bypassing SqlRepo validation, the DB rejects a bad stage."""
    t = sqldb.ModelVersionRow.__tablename__
    with repo.engine.begin() as conn:
        with pytest.raises(sa.exc.IntegrityError):
            conn.exec_driver_sql(
                f"INSERT INTO {t} (model_id, version_id, artifact_hash, stage, validated, "
                f"created_ts, updated_ts) VALUES ('a','b','{HASH}','bogus',0,1.0,1.0)")


def test_dataset_roundtrip(repo):
    repo.upsert_dataset(_ds())
    got = repo.get_dataset("d1")
    assert got["schema_json"] == {"a": "str", "b": "int"}
    assert got["rows"] == 10 and got["name"] == "clicks" and got["content_hash"] == HASH
    got["schema_json"]["a"] = "mutated"
    assert repo.get_dataset("d1")["schema_json"]["a"] == "str"
    # canonical JSON in storage: sorted keys, compact
    t = sqldb.DatasetVersionRow.__tablename__
    with repo.engine.connect() as conn:
        raw = conn.exec_driver_sql(f"SELECT schema_json FROM {t}").scalar()
    assert raw == '{"a":"str","b":"int"}'
    with pytest.raises(NotFound):
        repo.get_dataset("missing")


def test_count_covers_models(repo):
    assert repo.count() == 0
    for i in range(3):
        repo.upsert_model(_rec(version_id=f"v{i}"))
    repo.upsert_model(_rec(model_id="other"))
    assert repo.count() == 4


# --------------------------------------------------------------- validation
@pytest.mark.parametrize("field,value", [
    ("model_id", ""), ("version_id", ""), ("model_id", "a\x00b"), ("version_id", "a\nb"),
    ("model_id", "x" * 257), ("version_id", "x" * 257),
    ("artifact_hash", "A" * 64), ("artifact_hash", "a" * 63), ("artifact_hash", "g" * 64),
    ("artifact_hash", 5), ("created_ts", float("nan")), ("created_ts", float("inf")),
    ("created_ts", True), ("created_ts", "1.0"), ("updated_ts", float("-inf")),
    ("updated_ts", False), ("validated", "yes"), ("dataset_version", 3),
])
def test_model_validation_table(repo, field, value):
    with pytest.raises(ValidationFailed):
        repo.upsert_model(_rec(**{field: value}))
    assert repo.count() == 0


@pytest.mark.parametrize("field,value", [
    ("rows", -1), ("rows", True), ("rows", 1.5), ("version_id", ""), ("name", ""),
    ("content_hash", "zz"), ("schema_json", ["a"]), ("schema_json", {"a": {1, 2}}),
    ("schema_json", "{}"),
])
def test_dataset_validation_table(repo, field, value):
    with pytest.raises(ValidationFailed):
        repo.upsert_dataset(_ds(**{field: value}))
    with pytest.raises(NotFound):
        repo.get_dataset("d1")


# ---------------------------------------------------------------- injection
def test_sql_injection_is_inert(repo):
    evil = "x'; DROP TABLE model_version; --"
    repo.upsert_model(_rec(model_id=evil, version_id="v'\" OR 1=1"))
    got = repo.get_model(evil)
    assert len(got) == 1 and got[0]["model_id"] == evil and got[0]["version_id"] == "v'\" OR 1=1"
    assert sqldb.ModelVersionRow.__tablename__ in _tables(repo.engine.url.render_as_string())
    assert repo.get_model("x' OR '1'='1") == []
    assert repo.count() == 1


# --------------------------------------------------------------- concurrency
def _hammer(repos, per_thread=25, threads=8):
    errors = []

    def work(i):
        r = repos[i % len(repos)]
        try:
            for j in range(per_thread):
                r.upsert_model(_rec(model_id=f"m{i}", version_id=f"v{j}", created=float(j)))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    return errors


def test_concurrent_writes_one_repo(repo):
    assert _hammer([repo]) == []
    assert repo.count() == 200


def test_concurrent_writes_two_repos_same_file(migrated):
    a, b = sqldb.SqlRepo(migrated), sqldb.SqlRepo(migrated)
    try:
        assert _hammer([a, b]) == []
        assert a.count() == 200 and b.count() == 200
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------- persistence
def test_persistence_and_close_semantics(migrated):
    r1 = sqldb.SqlRepo(migrated)
    r1.upsert_model(_rec())
    r1.upsert_dataset(_ds())
    r1.close()
    r1.close()  # idempotent
    with pytest.raises((ValidationFailed, RuntimeError)):
        r1.get_model("m")
    with pytest.raises((ValidationFailed, RuntimeError)):
        r1.upsert_model(_rec(version_id="v9"))
    r2 = sqldb.SqlRepo(migrated)
    try:
        assert r2.count() == 1 and r2.get_model("m")[0]["version_id"] == "v1"
        assert r2.get_dataset("d1")["rows"] == 10
    finally:
        r2.close()
