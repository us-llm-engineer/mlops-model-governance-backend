"""The active policy gates production transitions and rollback in the real, fully-wired app.

Written before the wiring existed. Before this, ModelRegistry was built without a decision
point, so a model with accuracy 0.5 reached production under an active policy requiring 0.9.
Everything here goes through build_default_wiring + HTTP + a real SQLite file.
"""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))
warnings.simplefilter("ignore")

from fastapi.testclient import TestClient

from mlops.kernel import PolicyDenied
from mlops.policy_engine import PolicyBundle, Rule
from mlops.policy_store import PolicyStore, PolicyStoreGate
from mlops.audit import ChainedAuditStore
from mlops.svc.app import build_default_wiring, create_app
from mlops.svc.settings import Settings, load_settings

ADMIN = {"Authorization": "Bearer tk-admin-0000000000001"}
TOKENS = {"tk-admin-0000000000001": {"name": "root", "role": "admin"}}


def make(tmp_path, **extra):
    tmp_path.mkdir(exist_ok=True)
    s = Settings(audit_secret="enforce-secret-0123456789", tokens=TOKENS,
                 db_path=str(tmp_path / "m.db"), _env_file=None, **extra)
    wiring = build_default_wiring(s)
    return TestClient(create_app(s, **wiring)), wiring


def rule(rid, kind, params, severity="block"):
    return {"id": rid, "version": 1, "kind": kind, "severity": severity, "params": params}


def set_policy(client, rules, version=1):
    assert client.post("/v1/policy/publish", json={"name": "gate", "version": 1, "rules": rules},
                       headers=ADMIN).status_code == 201
    assert client.post(f"/v1/policy/{version}/activate", json={"human_approved_by": "root"},
                       headers=ADMIN).status_code == 200


def staged(client, model_id="m", version_id="1"):
    """Register, validate and stage a model version; returns the production-transition URL."""
    assert client.post("/v1/models", json={"model_id": model_id, "version_id": version_id,
                                            "artifact_hash": "b" * 64}, headers=ADMIN).status_code == 201
    base = f"/v1/models/{model_id}/versions/{version_id}"
    assert client.post(f"{base}/validate", json={"evidence": {"ok": True}}, headers=ADMIN).status_code == 200
    assert client.post(f"{base}/transition", json={"to_stage": "staging"}, headers=ADMIN).status_code == 200
    return f"{base}/transition"


def to_prod(client, url, context=None):
    body = {"to_stage": "production"}
    if context is not None:
        body["context"] = context
    return client.post(url, json=body, headers=ADMIN)


def stage_of(client, model_id, version_id):
    rows = client.get(f"/v1/models/{model_id}", headers=ADMIN).json()
    return next(r["stage"] for r in rows if r["version_id"] == version_id)


MIN_ACC = rule("acc", "min_metric", {"metric": "accuracy", "min": 0.9, "action": "stage.production"})


# ------------------------------------------------------------------ the gate itself
def test_gate_raises_on_deny_returns_decision_on_allow_and_shares_the_audit():
    audit = ChainedAuditStore(b"gate-secret")
    store = PolicyStore(audit)
    store.publish("a", PolicyBundle([Rule("f", 1, "flag_true", {"key": "go"}, "block")], "g", 1))
    store.activate("a", 1)
    gate = PolicyStoreGate(store)
    assert gate.audit is audit
    assert gate.enforce("stage.production", {"go": True}, actor="a").allow is True
    with pytest.raises(PolicyDenied) as e:
        gate.enforce("stage.production", {"go": False}, actor="a")
    assert "Rule f (flag_true) failed" in str(e.value)


def test_gate_with_no_active_policy_fails_closed():
    gate = PolicyStoreGate(PolicyStore(ChainedAuditStore(b"gate-secret")))
    with pytest.raises(PolicyDenied):
        gate.enforce("stage.production", {}, actor="a")


# ------------------------------------------------------------------ through the real app
def test_settings_default_enforces_and_env_can_turn_it_off():
    assert Settings(audit_secret="s" * 20, tokens=TOKENS, _env_file=None).policy_enforce_transitions is True
    base = {"MLOPS_AUDIT_SECRET": "s" * 20, "MLOPS_TOKENS": '{"tk-admin-00001": {"name": "r", "role": "admin"}}'}
    assert load_settings(None, env=dict(base)).policy_enforce_transitions is True
    for off in ("false", "0", "no"):
        assert load_settings(None, env={**base, "MLOPS_POLICY_ENFORCE_TRANSITIONS": off}).policy_enforce_transitions is False
    assert load_settings(None, env={**base, "MLOPS_POLICY_ENFORCE_TRANSITIONS": "true"}).policy_enforce_transitions is True


def test_failing_policy_blocks_production_and_leaves_the_stage_unchanged(tmp_path):
    client, wiring = make(tmp_path)
    url = staged(client)
    set_policy(client, [MIN_ACC])
    r = to_prod(client, url, {"accuracy": 0.5})
    assert r.status_code == 403 and r.json()["error"]["code"] == "policy_denied"
    assert stage_of(client, "m", "1") == "staging"
    denied = [e for e in wiring["audit"].export() if e["action"] == "policy.stage.production"]
    assert denied and denied[-1]["decision"] == "deny" and denied[-1]["resource"] == "m"


def test_passing_policy_allows_production(tmp_path):
    client, _ = make(tmp_path)
    url = staged(client)
    set_policy(client, [MIN_ACC])
    assert to_prod(client, url, {"accuracy": 0.95}).status_code == 200
    assert stage_of(client, "m", "1") == "production"


def test_missing_metric_in_the_context_is_a_denial(tmp_path):
    client, _ = make(tmp_path)
    url = staged(client)
    set_policy(client, [MIN_ACC])
    assert to_prod(client, url).status_code == 403
    assert to_prod(client, url, {"accuracy": True}).status_code == 403


def test_no_active_policy_blocks_production_fail_closed(tmp_path):
    client, _ = make(tmp_path)
    url = staged(client)
    r = to_prod(client, url, {"accuracy": 1.0})
    assert r.status_code == 403 and r.json()["error"]["code"] == "policy_denied"
    assert stage_of(client, "m", "1") == "staging"


def test_a_rule_scoped_to_another_action_does_not_block(tmp_path):
    client, _ = make(tmp_path)
    url = staged(client)
    set_policy(client, [rule("d", "flag_true", {"key": "approved", "action": "deploy"})])
    assert to_prod(client, url).status_code == 200


def test_only_production_is_gated_staging_is_not(tmp_path):
    client, _ = make(tmp_path)
    assert client.post("/v1/models", json={"model_id": "m", "version_id": "1", "artifact_hash": "b" * 64},
                       headers=ADMIN).status_code == 201
    client.post("/v1/models/m/versions/1/validate", json={"evidence": {"ok": True}}, headers=ADMIN)
    set_policy(client, [MIN_ACC])
    r = client.post("/v1/models/m/versions/1/transition", json={"to_stage": "staging"}, headers=ADMIN)
    assert r.status_code == 200


def test_server_sets_role_client_cannot_spoof_it(tmp_path):
    client, _ = make(tmp_path)
    url = staged(client)
    set_policy(client, [rule("r", "role_in", {"key": "role", "roles": ["admin"], "action": "stage.production"})])
    # the caller is an admin: a false claim of another role must not deny them...
    assert to_prod(client, url, {"role": "nobody"}).status_code == 200
    client2, _ = make(tmp_path / "b")
    url2 = staged(client2)
    set_policy(client2, [rule("r", "role_in", {"key": "role", "roles": ["operator"], "action": "stage.production"})])
    # ...and claiming a permitted role must not admit them.
    assert to_prod(client2, url2, {"role": "operator"}).status_code == 403


def test_resource_is_the_model_id_and_cannot_be_overridden_by_the_client(tmp_path):
    client, _ = make(tmp_path)
    url = staged(client, "protected-1")
    url_other = staged(client, "other-1")
    set_policy(client, [rule("a", "flag_true", {"key": "approved", "action": "stage.production", "resource": "protected-*"})])
    assert to_prod(client, url_other).status_code == 200                      # rule out of scope
    assert to_prod(client, url, {"resource": "other-1"}).status_code == 403   # client's resource ignored
    assert to_prod(client, url, {"approved": True}).status_code == 200


def test_rollback_is_gated_too(tmp_path):
    client, _ = make(tmp_path)
    flag = rule("a", "flag_true", {"key": "approved", "action": "stage.production"})
    u1 = staged(client, "m", "1")
    u2 = staged(client, "m", "2")
    set_policy(client, [flag])
    assert to_prod(client, u1, {"approved": True}).status_code == 200
    assert to_prod(client, u2, {"approved": True}).status_code == 200
    denied = client.post("/v1/models/m/rollback", json={}, headers=ADMIN)
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "policy_denied"
    assert stage_of(client, "m", "2") == "production"
    ok = client.post("/v1/models/m/rollback", json={"context": {"approved": True}}, headers=ADMIN)
    assert ok.status_code == 200 and stage_of(client, "m", "1") == "production"


def test_switching_enforcement_off_restores_ungated_transitions(tmp_path):
    client, _ = make(tmp_path, policy_enforce_transitions=False)
    url = staged(client)
    set_policy(client, [MIN_ACC])
    assert to_prod(client, url, {"accuracy": 0.1}).status_code == 200


def test_decision_point_is_wired_into_the_default_registry(tmp_path):
    _, wiring = make(tmp_path)
    assert isinstance(wiring["registry"].decision_point, PolicyStoreGate)
    _, off = make(tmp_path / "x", policy_enforce_transitions=False)
    assert off["registry"].decision_point is None
