"""Policy versions survive a restart (PolicyStore write-through to OpsRepo).

Written before the code. Before this, PolicyStore was in memory only: OpsRepo.upsert_policy_version
had no production caller, and rebuilding the app on the same DB file turned /v1/policy/active
from 200 into 404 and emptied /v1/policy/versions.
"""
import json
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))
warnings.simplefilter("ignore")

from fastapi.testclient import TestClient

from mlops.audit import ChainedAuditStore
from mlops.kernel import Conflict, IntegrityError
from mlops.policy_engine import PolicyBundle, Rule
from mlops.policy_store import PolicyStore
from mlops.sqldb_ops import OpsRepo
from mlops.svc.app import build_default_wiring, create_app
from mlops.svc.settings import Settings

ADMIN = {"Authorization": "Bearer tk-admin-0000000000001"}
TOKENS = {"tk-admin-0000000000001": {"name": "root", "role": "admin"}}


def boot(db_path):
    s = Settings(audit_secret="persist-secret-0123456789", tokens=TOKENS, db_path=str(db_path), _env_file=None)
    wiring = build_default_wiring(s)
    return TestClient(create_app(s, **wiring)), wiring


def rule(rid, min_):
    return {"id": rid, "version": 1, "kind": "min_metric", "severity": "block",
            "params": {"metric": "accuracy", "min": min_, "action": "stage.production"}}


def publish(client, rules, note=""):
    body = {"name": "gate", "version": 1, "rules": rules}
    if note:
        body["note"] = note
    r = client.post("/v1/policy/publish", json=body, headers=ADMIN)
    assert r.status_code == 201, r.text
    return r.json()["version"]


def activate(client, version, approver="root"):
    return client.post(f"/v1/policy/{version}/activate", json={"human_approved_by": approver}, headers=ADMIN)


def decide(client, acc):
    return client.post("/v1/policy/decide", json={"action": "stage.production", "context": {"accuracy": acc}},
                       headers=ADMIN).json()


# ------------------------------------------------------------------ through the real app
def test_active_policy_and_history_survive_a_restart(tmp_path):
    db = tmp_path / "m.db"
    c1, _ = boot(db)
    v1 = publish(c1, [rule("acc", 0.7)], note="first")
    v2 = publish(c1, [rule("acc", 0.9)], note="stricter")
    assert activate(c1, v1).status_code == 200
    assert activate(c1, v2).status_code == 200
    before = decide(c1, 0.8)
    hist_before = c1.get("/v1/policy/versions", headers=ADMIN).json()

    c2, _ = boot(db)                                       # same file, fresh process state
    active = c2.get("/v1/policy/active", headers=ADMIN)
    assert active.status_code == 200 and active.json() == {"version": 2, "name": "gate"}
    assert decide(c2, 0.8) == before                       # same verdict, reasons, hash and version
    assert before["allow"] is False and before["version"] == 2
    assert c2.get("/v1/policy/versions", headers=ADMIN).json() == hist_before
    assert c2.get("/v1/policy/versions/1", headers=ADMIN).status_code == 200


def test_publishing_without_activating_does_not_change_the_active_policy_after_a_restart(tmp_path):
    db = tmp_path / "m.db"
    c1, wiring = boot(db)
    activate(c1, publish(c1, [rule("acc", 0.9)]))
    publish(c1, [rule("acc", 0.1)])                       # a much looser draft, never activated
    assert [r["active"] for r in wiring["ops_repo"].list_policy_versions()] == [True, False]
    c2, _ = boot(db)
    assert c2.get("/v1/policy/active", headers=ADMIN).json()["version"] == 1
    assert decide(c2, 0.5)["allow"] is False


def test_version_numbers_continue_after_a_restart(tmp_path):
    db = tmp_path / "m.db"
    c1, _ = boot(db)
    publish(c1, [rule("acc", 0.7)])
    publish(c1, [rule("acc", 0.9)])
    c2, _ = boot(db)
    assert publish(c2, [rule("acc", 0.95)]) == 3


def test_loosening_guard_still_applies_against_the_reloaded_active_policy(tmp_path):
    db = tmp_path / "m.db"
    c1, _ = boot(db)
    v_loose = publish(c1, [rule("acc", 0.7)])
    v_strict = publish(c1, [rule("acc", 0.9)])
    activate(c1, v_loose)
    activate(c1, v_strict)
    c2, _ = boot(db)
    assert c2.post(f"/v1/policy/{v_loose}/activate", json={}, headers=ADMIN).status_code == 403
    assert activate(c2, v_loose).status_code == 200


def test_enforcement_survives_a_restart(tmp_path):
    db = tmp_path / "m.db"
    c1, _ = boot(db)
    activate(c1, publish(c1, [rule("acc", 0.9)]))
    c2, _ = boot(db)
    base = "/v1/models/m/versions/1"
    assert c2.post("/v1/models", json={"model_id": "m", "version_id": "1", "artifact_hash": "b" * 64},
                   headers=ADMIN).status_code == 201
    c2.post(f"{base}/validate", json={"evidence": {"ok": True}}, headers=ADMIN)
    c2.post(f"{base}/transition", json={"to_stage": "staging"}, headers=ADMIN)
    assert c2.post(f"{base}/transition", json={"to_stage": "production", "context": {"accuracy": 0.5}},
                   headers=ADMIN).status_code == 403


def test_repo_rows_have_exactly_one_active_and_a_parseable_bundle(tmp_path):
    db = tmp_path / "m.db"
    c1, wiring = boot(db)
    v1 = publish(c1, [rule("acc", 0.7)], note="n1")
    v2 = publish(c1, [rule("acc", 0.9)])
    activate(c1, v1)
    activate(c1, v2)
    rows = wiring["ops_repo"].list_policy_versions()
    assert [(r["version"], r["active"], r["note"]) for r in rows] == [(1, False, "n1"), (2, True, "")]
    bundle = wiring["policy_store"].get(2)
    assert json.loads(rows[1]["bundle_json"]) == bundle.to_dict()


def test_in_memory_database_keeps_the_store_in_memory(tmp_path):
    s = Settings(audit_secret="persist-secret-0123456789", tokens=TOKENS, db_path=":memory:", _env_file=None)
    wiring = build_default_wiring(s)
    assert wiring["ops_repo"] is None and wiring["policy_store"].repo is None
    c = TestClient(create_app(s, **wiring))
    assert activate(c, publish(c, [rule("acc", 0.9)])).status_code == 200


# ------------------------------------------------------------------ store-level behaviour
def bundle(min_=0.9):
    return PolicyBundle([Rule("acc", 1, "min_metric", {"metric": "accuracy", "min": min_}, "block")], "g", 1)


class FlakyRepo:
    """Real OpsRepo semantics are covered above; this one fails on demand."""
    def __init__(self, inner):
        self.inner, self.fail = inner, False

    def upsert_policy_version(self, record):
        if self.fail:
            raise RuntimeError("disk full")
        self.inner.upsert_policy_version(record)

    def list_policy_versions(self):
        return self.inner.list_policy_versions()


def flaky(tmp_path):
    from mlops.sqldb import upgrade
    url = f"sqlite:///{tmp_path / 'r.db'}"
    upgrade(url, "head")
    return FlakyRepo(OpsRepo(url)), url


def test_a_failed_publish_write_changes_nothing(tmp_path):
    repo, _ = flaky(tmp_path)
    st = PolicyStore(ChainedAuditStore(b"s"), repo=repo)
    repo.fail = True
    with pytest.raises(RuntimeError):
        st.publish("a", bundle())
    assert st.history() == []
    repo.fail = False
    assert st.publish("a", bundle()) == 1                   # the version number was not consumed


def test_a_failed_activate_write_leaves_the_active_version_alone(tmp_path):
    repo, _ = flaky(tmp_path)
    st = PolicyStore(ChainedAuditStore(b"s"), repo=repo)
    st.publish("a", bundle(0.7))
    st.publish("a", bundle(0.9))
    st.activate("a", 1)
    repo.fail = True
    with pytest.raises(RuntimeError):
        st.activate("a", 2)
    assert st.active()[0] == 1
    repo.fail = False
    st.activate("a", 2)
    assert st.active()[0] == 2


def test_rollback_right_after_a_restart_has_no_previous_version(tmp_path):
    repo, url = flaky(tmp_path)
    st = PolicyStore(ChainedAuditStore(b"s"), repo=repo)
    st.publish("a", bundle(0.7))
    st.publish("a", bundle(0.9))
    st.activate("a", 1)
    st.activate("a", 2)
    reloaded = PolicyStore(ChainedAuditStore(b"s"), repo=OpsRepo(url))
    assert reloaded.active()[0] == 2
    with pytest.raises(Conflict):
        reloaded.rollback("a")


def test_a_corrupt_stored_row_fails_loudly_instead_of_starting_empty(tmp_path):
    repo, url = flaky(tmp_path)
    st = PolicyStore(ChainedAuditStore(b"s"), repo=repo)
    st.publish("a", bundle())
    inner = OpsRepo(url)
    rec = inner.list_policy_versions()[0]
    inner.upsert_policy_version({**rec, "bundle_json": "{not json"})
    with pytest.raises(IntegrityError):
        PolicyStore(ChainedAuditStore(b"s"), repo=OpsRepo(url))
    bad_rule = json.dumps({"name": "g", "version": 1, "rules": [{"id": "x", "version": 1, "kind": "nope", "params": {}}]})
    inner.upsert_policy_version({**rec, "bundle_json": bad_rule})
    with pytest.raises(IntegrityError):
        PolicyStore(ChainedAuditStore(b"s"), repo=OpsRepo(url))
