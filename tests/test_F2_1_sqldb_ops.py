"""F2 round 2 - mlops.sqldb_ops contract (OpsRepo: policy versions, incidents, drift baselines).

Written BEFORE the code by the test author; the weaker implementer codes against this frozen
file and must not edit it. Covers section 1 of the API contract only -- not
manifest_io/audit_stream/cyclonedx_sbom/cli_ops/wiring (sections 2-6), which get their own
frozen suites elsewhere.

Migration round-trip checks use raw sqlite3 introspection (sqlite_master), never the ORM,
so the test does not trust the code path it is verifying.
"""
import os
import sqlite3
import sys

import pytest
import sqlalchemy as sa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.kernel import NotFound, ValidationFailed  # noqa: E402
import mlops.sqldb as sqldb  # noqa: E402
import mlops.sqldb_ops as sqldb_ops  # noqa: E402


def _url(tmp_path, name="ops.db"):
    return f"sqlite:///{tmp_path}/{name}"


def _sqlite_tables(tmp_path, name="ops.db"):
    """Raw sqlite3 introspection -- deliberately bypasses the ORM/repo under test."""
    con = sqlite3.connect(f"{tmp_path}/{name}")
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {r[0] for r in rows}
    finally:
        con.close()


def _policy(version=1, name="p", active=False, **kw):
    r = {"version": version, "name": name, "bundle_json": "{}", "note": "",
         "published_ts": 1.0, "active": active}
    r.update(kw)
    return r


def _incident(incident_id="i1", severity="P1", status="open", **kw):
    r = {"incident_id": incident_id, "title": "t", "severity": severity, "status": status,
         "opened_ts": 1.0, "signal_json": "{}"}
    r.update(kw)
    return r


@pytest.fixture
def migrated(tmp_path):
    url = _url(tmp_path)
    sqldb.upgrade(url)
    return url


@pytest.fixture
def repo(migrated):
    r = sqldb_ops.OpsRepo(migrated)
    yield r
    r.close()


# --------------------------------------------------------------- migrations
def test_upgrade_head_twice_is_idempotent_and_creates_ops_tables(tmp_path):
    url = _url(tmp_path)
    sqldb.upgrade(url, "head")
    first = sqldb.current_revision(url)
    sqldb.upgrade(url, "head")  # must be a no-op, not an error
    second = sqldb.current_revision(url)
    assert first == second
    names = _sqlite_tables(tmp_path)
    assert {"policy_version", "incident", "drift_baseline"} <= names


def test_downgrade_to_0002_drops_ops_tables_then_upgrade_head_restores(tmp_path):
    url = _url(tmp_path)
    sqldb.upgrade(url, "head")
    assert {"policy_version", "incident", "drift_baseline"} <= _sqlite_tables(tmp_path)
    sqldb.downgrade(url, "0002")
    names = _sqlite_tables(tmp_path)
    assert not ({"policy_version", "incident", "drift_baseline"} & names)
    assert "dataset_version" in names  # 0002's own table must survive this downgrade
    sqldb.upgrade(url, "head")
    names = _sqlite_tables(tmp_path)
    assert {"policy_version", "incident", "drift_baseline"} <= names


# ------------------------------------------------------------- CHECK constraints
def test_incident_check_constraints_enforced_by_database_bypassing_repo(repo):
    """Belt and braces: even bypassing OpsRepo validation, the DB itself rejects bad enums."""
    t = sqldb_ops.IncidentRow.__tablename__
    with repo.engine.begin() as conn:
        with pytest.raises(sa.exc.IntegrityError):
            conn.exec_driver_sql(
                f"INSERT INTO {t} (incident_id, title, severity, status, opened_ts, signal_json) "
                f"VALUES ('x','t','BOGUS','open',1.0,'{{}}')")
    with repo.engine.begin() as conn:
        with pytest.raises(sa.exc.IntegrityError):
            conn.exec_driver_sql(
                f"INSERT INTO {t} (incident_id, title, severity, status, opened_ts, signal_json) "
                f"VALUES ('y','t','P1','bogus',1.0,'{{}}')")


def test_incident_bad_severity_via_repo_is_validation_failed(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_incident(_incident(severity="BOGUS"))
    with pytest.raises(NotFound):
        repo.get_incident("i1")


def test_incident_bad_status_via_repo_is_validation_failed(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_incident(_incident(status="closed"))
    with pytest.raises(NotFound):
        repo.get_incident("i1")


# ------------------------------------------------------------- policy version
def test_policy_version_insert_and_get(repo):
    repo.upsert_policy_version(_policy(version=1, name="alpha", note="n1"))
    got = repo.get_policy_version(1)
    assert got["name"] == "alpha" and got["note"] == "n1" and got["active"] is False


def test_policy_version_update_in_place(repo):
    repo.upsert_policy_version(_policy(version=1, name="alpha"))
    repo.upsert_policy_version(_policy(version=1, name="beta"))
    assert repo.get_policy_version(1)["name"] == "beta"
    assert len(repo.list_policy_versions()) == 1


def test_policy_version_must_be_int_not_bool(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_policy_version(_policy(version=True))
    assert repo.list_policy_versions() == []


def test_policy_version_must_be_geq_1(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_policy_version(_policy(version=0))
    with pytest.raises(ValidationFailed):
        repo.upsert_policy_version(_policy(version=-1))
    assert repo.list_policy_versions() == []


def test_policy_version_active_flip_single_active_invariant_same_call(repo):
    repo.upsert_policy_version(_policy(version=1, active=True))
    repo.upsert_policy_version(_policy(version=2, active=True))
    all_v = {r["version"]: r["active"] for r in repo.list_policy_versions()}
    assert all_v == {1: False, 2: True}


def test_policy_version_setting_active_false_does_not_disturb_others(repo):
    repo.upsert_policy_version(_policy(version=1, active=True))
    repo.upsert_policy_version(_policy(version=2, active=False))
    all_v = {r["version"]: r["active"] for r in repo.list_policy_versions()}
    assert all_v == {1: True, 2: False}


def test_get_policy_version_not_found(repo):
    with pytest.raises(NotFound):
        repo.get_policy_version(99)


def test_list_policy_versions_ordering(repo):
    repo.upsert_policy_version(_policy(version=3))
    repo.upsert_policy_version(_policy(version=1))
    repo.upsert_policy_version(_policy(version=2))
    assert [r["version"] for r in repo.list_policy_versions()] == [1, 2, 3]


def test_policy_version_note_and_bundle_roundtrip(repo):
    repo.upsert_policy_version(_policy(version=1, note="hello", bundle_json='{"a":1}'))
    got = repo.get_policy_version(1)
    assert got["note"] == "hello"
    assert got["bundle_json"] == '{"a":1}'


# ------------------------------------------------------------------ incidents
def test_incident_insert_and_get(repo):
    repo.upsert_incident(_incident(incident_id="i1", title="disk full"))
    got = repo.get_incident("i1")
    assert got["title"] == "disk full" and got["severity"] == "P1" and got["status"] == "open"


def test_incident_update_in_place(repo):
    repo.upsert_incident(_incident(incident_id="i1", status="open"))
    repo.upsert_incident(_incident(incident_id="i1", status="resolved"))
    assert repo.get_incident("i1")["status"] == "resolved"


def test_get_incident_not_found(repo):
    with pytest.raises(NotFound):
        repo.get_incident("missing")


def test_count_open_incidents_by_severity_all_zero_when_none(repo):
    assert repo.count_open_incidents_by_severity() == {"P0": 0, "P1": 0, "P2": 0, "P3": 0}


def test_count_open_incidents_by_severity_mixed(repo):
    repo.upsert_incident(_incident(incident_id="a", severity="P0", status="open"))
    repo.upsert_incident(_incident(incident_id="b", severity="P0", status="open"))
    repo.upsert_incident(_incident(incident_id="c", severity="P0", status="resolved"))
    repo.upsert_incident(_incident(incident_id="d", severity="P2", status="open"))
    repo.upsert_incident(_incident(incident_id="e", severity="P3", status="resolved"))
    counts = repo.count_open_incidents_by_severity()
    assert counts == {"P0": 2, "P1": 0, "P2": 1, "P3": 0}


def test_count_open_incidents_by_severity_updates_after_resolve(repo):
    repo.upsert_incident(_incident(incident_id="a", severity="P1", status="open"))
    assert repo.count_open_incidents_by_severity()["P1"] == 1
    repo.upsert_incident(_incident(incident_id="a", severity="P1", status="resolved"))
    assert repo.count_open_incidents_by_severity()["P1"] == 0


# -------------------------------------------------------------- drift baseline
def test_drift_baseline_insert_and_get(repo):
    ref = [float(i) for i in range(30)]
    repo.upsert_drift_baseline("latency", ref)
    assert repo.get_drift_baseline("latency") == ref


def test_drift_baseline_name_empty_rejected(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_drift_baseline("", [float(i) for i in range(30)])


def test_drift_baseline_name_control_chars_rejected(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_drift_baseline("bad\x00name", [float(i) for i in range(30)])


def test_drift_baseline_name_too_long_rejected(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_drift_baseline("x" * 257, [float(i) for i in range(30)])


def test_drift_baseline_reference_below_minimum_rejected(repo):
    with pytest.raises(ValidationFailed):
        repo.upsert_drift_baseline("latency", [float(i) for i in range(29)])
    with pytest.raises(NotFound):
        repo.get_drift_baseline("latency")


def test_drift_baseline_reference_exactly_30_accepted(repo):
    ref = [float(i) for i in range(30)]
    repo.upsert_drift_baseline("latency", ref)
    assert len(repo.get_drift_baseline("latency")) == 30


def test_drift_baseline_reference_nan_rejected(repo):
    ref = [float(i) for i in range(29)] + [float("nan")]
    with pytest.raises(ValidationFailed):
        repo.upsert_drift_baseline("latency", ref)


def test_drift_baseline_reference_inf_rejected(repo):
    ref = [float(i) for i in range(29)] + [float("inf")]
    with pytest.raises(ValidationFailed):
        repo.upsert_drift_baseline("latency", ref)


def test_drift_baseline_unique_name_upserts_in_place(repo):
    ref1 = [float(i) for i in range(30)]
    ref2 = [float(i) * 2 for i in range(30)]
    repo.upsert_drift_baseline("latency", ref1)
    repo.upsert_drift_baseline("latency", ref2)
    assert repo.get_drift_baseline("latency") == ref2
    with repo.engine.connect() as conn:
        n = conn.exec_driver_sql(
            f"SELECT COUNT(*) FROM {sqldb_ops.DriftBaselineRow.__tablename__}").scalar()
    assert n == 1


def test_get_drift_baseline_not_found(repo):
    with pytest.raises(NotFound):
        repo.get_drift_baseline("missing")
