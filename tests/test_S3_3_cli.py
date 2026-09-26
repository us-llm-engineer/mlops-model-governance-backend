"""S3.3 CLI contract (frozen before code). Module under test: mlops.svc.cli.

Runner facts (typer 0.27.1 / bundled click 8.4.2): stdout and stderr are captured
separately (result.stdout / result.stderr); result.output is the mixed view.
Assertions on streams therefore use stdout/stderr explicitly.
"""
import json
import re

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from mlops.svc.app import create_app
from mlops.svc.cli import create_cli
from mlops.svc.sdk import MlopsClient
from mlops.svc.settings import Settings

HASH = "a" * 64
ADMIN, OPER, VIEW = "tk-admin-00001", "tk-oper-000001", "tk-view-000001"
TOKENS = {
    ADMIN: {"name": "alice", "role": "admin"},
    OPER: {"name": "bob", "role": "operator"},
    VIEW: {"name": "carol", "role": "viewer"},
}
runner = CliRunner()


@pytest.fixture
def api():
    settings = Settings(audit_secret="s" * 20, tokens=TOKENS, _env_file=None)
    return TestClient(create_app(settings))


def make_cli(api, token=ADMIN):
    return create_cli(lambda: MlopsClient(client=api, token=token))


def run(api, args, token=ADMIN):
    return runner.invoke(make_cli(api, token), args)


def no_traceback(r):
    assert "Traceback" not in r.output and "File \"" not in r.output


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return str(p)


RULES = {
    "name": "p", "version": 1,
    "rules": [{"id": "r1", "version": 1, "kind": "min_metric",
               "params": {"metric": "acc", "min": 0.9}, "severity": "block"}],
}
CLEAN_POD = {
    "apiVersion": "v1", "kind": "Pod",
    "metadata": {"name": "ok", "namespace": "prod"},
    "spec": {"containers": [{
        "name": "c", "image": "registry.local/app:1.2.3",
        "resources": {"requests": {"cpu": "1"}, "limits": {"cpu": "1"}},
        "livenessProbe": {"httpGet": {"path": "/", "port": 80}},
        "readinessProbe": {"httpGet": {"path": "/", "port": 80}},
        "securityContext": {"readOnlyRootFilesystem": True},
    }]},
}


def bad_pod():
    pod = json.loads(json.dumps(CLEAN_POD))
    pod["spec"]["containers"][0]["securityContext"]["privileged"] = True
    return pod


def test_health_json(api):
    r = run(api, ["health"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["status"] == "ok"


def test_models_register_transition_get(api):
    assert run(api, ["models", "register", "M", "v1", HASH]).exit_code == 0
    r = run(api, ["models", "transition", "M", "v1", "staging"])
    assert r.exit_code == 0, r.output
    r = run(api, ["models", "get", "M"])
    assert r.exit_code == 0, r.output
    assert "staging" in r.stdout
    json.loads(r.stdout)


def test_api_error_single_line_on_stderr(api):
    run(api, ["models", "register", "M", "v1", HASH])
    r = run(api, ["models", "transition", "M", "v1", "production"])  # registered -> production is illegal
    assert r.exit_code == 1
    lines = r.stderr.strip().splitlines()
    assert len(lines) == 1 and re.match(r"^error: [A-Za-z_]+: .+", lines[0])
    assert r.stdout == ""
    no_traceback(r)


def test_not_found_error_code_present(api):
    r = run(api, ["models", "transition", "nope", "v9", "staging"])
    assert r.exit_code == 1 and r.stderr.startswith("error: ")
    assert len(r.stderr.strip().splitlines()) == 1 and r.stdout == ""


def test_forbidden_viewer_exit_1(api):
    r = run(api, ["models", "register", "M", "v1", HASH], token=VIEW)
    assert r.exit_code == 1 and r.stderr.startswith("error: ")
    no_traceback(r)


def test_bad_token_exit_1_and_never_printed(api):
    secret = "tk-totally-wrong-token-99"
    r = run(api, ["audit", "head"], token=secret)
    assert r.exit_code == 1 and r.stderr.startswith("error: ")
    assert secret not in r.stdout and secret not in r.stderr and secret not in r.output
    no_traceback(r)


@pytest.mark.parametrize("args", [
    ["bogus"],
    ["models", "register", "M", "v1"],
    ["models", "transition", "M", "v1", "sideways"],
    ["policy", "activate", "abc"],
    ["policy", "activate", "true"],
    ["policy", "activate", "-1"],
    ["policy", "decide", "promote"],
])
def test_usage_errors_exit_2(api, args):
    r = run(api, args)
    assert r.exit_code == 2
    no_traceback(r)


def test_policy_publish_activate_decide_active(api, tmp_path):
    f = write(tmp_path, "p.json", RULES)
    assert run(api, ["policy", "publish", f]).exit_code == 0
    assert run(api, ["policy", "activate", "1"]).exit_code == 0
    ok = run(api, ["policy", "decide", "promote", '{"acc":0.95}'])
    assert ok.exit_code == 0 and json.loads(ok.stdout)["allow"] is True
    no = run(api, ["policy", "decide", "promote", '{"acc":0.5}'])
    assert no.exit_code == 0 and json.loads(no.stdout)["allow"] is False
    act = run(api, ["policy", "active"])
    assert act.exit_code == 0 and json.loads(act.stdout)


@pytest.mark.parametrize("ctx", ["not json", "[1, 2]", "NaN", '{"acc": NaN}'])
def test_bad_context_json_exit_2(api, tmp_path, ctx):
    run(api, ["policy", "publish", write(tmp_path, "p.json", RULES)])
    run(api, ["policy", "activate", "1"])
    r = run(api, ["policy", "decide", "promote", ctx])
    assert r.exit_code == 2 and r.output.strip()
    no_traceback(r)


def test_lint_privileged_blocks(api, tmp_path):
    r = run(api, ["lint", write(tmp_path, "m.json", bad_pod())])
    assert r.exit_code == 1, r.output
    out = json.loads(r.stdout)
    assert any(f["rule_id"] == "privileged" for f in out["findings"])


def test_lint_clean_manifest_exit_0(api, tmp_path):
    r = run(api, ["lint", write(tmp_path, "m.json", CLEAN_POD)])
    assert r.exit_code == 0, r.output
    assert "findings" in json.loads(r.stdout)


def test_lint_bad_files_exit_2(api, tmp_path):
    big = write(tmp_path, "big.json", '{"a":"' + "x" * (5 * 1024 * 1024 + 1) + '"}')
    for path in (str(tmp_path / "missing.json"), write(tmp_path, "bad.json", "{oops"), big):
        r = run(api, ["lint", path])
        assert r.exit_code == 2 and r.output.strip(), path
        no_traceback(r)


def test_drift_check_report_and_short_window(api, tmp_path):
    ref = write(tmp_path, "ref.json", [i / 100 for i in range(100)])
    win = write(tmp_path, "win.json", [i / 40 for i in range(40)])
    r = run(api, ["drift", "check", ref, win])
    assert r.exit_code == 0, r.output
    assert "level" in json.loads(r.stdout)
    short = write(tmp_path, "short.json", [0.1] * 10)
    r = run(api, ["drift", "check", ref, short])
    assert r.exit_code == 1 and r.stderr.startswith("error: ")
    no_traceback(r)


def test_audit_head(api):
    run(api, ["models", "register", "M", "v1", HASH])
    r = run(api, ["audit", "head"])
    assert r.exit_code == 0, r.output
    head = json.loads(r.stdout)
    assert head["length"] >= 1 and head["signature"] and "anchor" in head


def test_default_factory_missing_env_exit_2(monkeypatch):
    from mlops.svc import cli
    for missing, other in (("MLOPS_URL", "MLOPS_TOKEN"), ("MLOPS_TOKEN", "MLOPS_URL")):
        monkeypatch.setenv(other, "x-value-1234567")
        monkeypatch.delenv(missing, raising=False)
        r = runner.invoke(cli.app, ["health"])
        assert r.exit_code == 2 and missing in r.output
        no_traceback(r)


def test_default_factory_builds_from_env(monkeypatch):
    from mlops.svc import cli, sdk
    seen = []

    class Rec:
        def __init__(self, *a, **k):
            seen.append(list(a) + list(k.values()))

    monkeypatch.setenv("MLOPS_URL", "http://mlops.example.invalid:9")
    monkeypatch.setenv("MLOPS_TOKEN", "tk-env-token-0001")
    monkeypatch.setattr(sdk, "MlopsClient", Rec)
    monkeypatch.setattr(cli, "MlopsClient", Rec, raising=False)
    cli.default_factory()
    assert "http://mlops.example.invalid:9" in seen[0]


def test_output_is_stable_sorted_json(api):
    run(api, ["models", "register", "M", "v1", HASH])
    r = run(api, ["models", "get", "M"])
    assert r.stdout == json.dumps(json.loads(r.stdout), indent=2, sort_keys=True) + "\n"


def test_help_lists_every_command(api):
    tree = {
        None: ["models", "policy", "lint", "drift", "audit", "health"],
        "models": ["register", "transition", "get"],
        "policy": ["publish", "activate", "decide", "active"],
        "drift": ["check"],
        "audit": ["head"],
    }
    for group, names in tree.items():
        r = run(api, ([group] if group else []) + ["--help"])
        assert r.exit_code == 0, (group, r.output)
        for n in names:
            assert n in r.stdout, (group, n)
