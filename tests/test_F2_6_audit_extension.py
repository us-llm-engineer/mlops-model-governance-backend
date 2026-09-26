"""F2.6: the real `audit_extension` in mlops.svc.extensions (contract sec. 6).

The router suite (test_F2_3_audit_router.py) mounts the router through its own local extension, so
without this file the production wiring function would have no test. Kept tiny on purpose: it only
proves the shipped extension mounts the audit routes with the real auth in front of them.
Written before the implementation and frozen with the rest.
"""
import functools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exec"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mlops.audit import ChainedAuditStore, ChainVerifier  # noqa: E402
from mlops.kernel import ValidationFailed  # noqa: E402
from mlops.svc import extensions as ext_mod  # noqa: E402
from mlops.svc.app import create_app  # noqa: E402
from mlops.svc.settings import Settings  # noqa: E402

SECRET = b"f2-6-extension-secret-000"
TOK = {"viewer": "tk-viewer-0001", "operator": "tk-operator-01", "admin": "tk-admin-00001"}


def _client(with_ext=True):
    tokens = {TOK[r]: {"name": r, "role": r} for r in TOK}
    settings = Settings(audit_secret="s" * 20, tokens=tokens, _env_file=None)
    audit = ChainedAuditStore(SECRET)
    for i in range(2):
        audit.append("alice", f"a{i}", f"r{i}", "allow", meta={"i": i})
    exts = [ext_mod.audit_extension(functools.partial(ChainVerifier, SECRET))] if with_ext else None
    app = create_app(settings, audit=audit, extensions=exts)
    return TestClient(app, raise_server_exceptions=False), audit, app


def H(role):
    return {"Authorization": "Bearer " + TOK[role]}


def test_audit_extension_returns_a_callable_taking_app_and_state():
    assert callable(ext_mod.audit_extension(functools.partial(ChainVerifier, SECRET)))


@pytest.mark.parametrize("bad", [None, "x", 3, True])
def test_audit_extension_rejects_a_non_callable_verifier_factory(bad):
    with pytest.raises(ValidationFailed):
        ext_mod.audit_extension(bad)


def test_extension_mounts_both_audit_routes_in_openapi():
    _, _, app = _client()
    paths = app.openapi()["paths"]
    assert "get" in paths["/v1/audit/tail"]
    assert "post" in paths["/v1/audit/verify"]


def test_routes_are_absent_without_the_extension():
    _, _, app = _client(with_ext=False)
    paths = app.openapi()["paths"]
    assert "/v1/audit/tail" not in paths and "/v1/audit/verify" not in paths


def test_verify_through_the_extension_is_authenticated_and_answers_honestly():
    c, audit, _ = _client()
    body = {"export": audit.export(), "head": audit.head()}
    assert c.post("/v1/audit/verify", json=body).status_code == 401
    assert c.post("/v1/audit/verify", json=body, headers=H("viewer")).status_code == 403
    ok = c.post("/v1/audit/verify", json=body, headers=H("operator"))
    assert ok.status_code == 200 and ok.json() == {"valid": True}
