"""S3.3 slim suite: mlops.svc.settings (Settings, load_settings, config_hash).
Frozen BEFORE the module exists. Contract: the API contract
(svc/settings.py). Hermetic: every load passes an explicit `env` mapping.
"""
import json
import os
import signal
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.kernel import ValidationFailed
from mlops.svc.settings import Settings, config_hash, load_settings

SECRET = "s3cr3t-value-abcdef-0123456789"
TOK1 = "tok-one-AAAA1111"
TOK2 = "tok-two-BBBB2222"
MIB = 1024 * 1024


def base(**over):
    cfg = {
        "audit_secret": SECRET,
        "tokens": {TOK1: {"name": "alice", "role": "admin"}},
    }
    cfg.update(over)
    return cfg


def write(tmp_path, cfg, name="cfg.json"):
    p = tmp_path / name
    p.write_text(cfg if isinstance(cfg, str) else json.dumps(cfg))
    return str(p)


def load(tmp_path, cfg, env=None):
    return load_settings(write(tmp_path, cfg), env={} if env is None else env)


class alarm:
    """Fail (not hang) if the body runs longer than `sec` seconds."""

    def __init__(self, sec=5):
        self.sec = sec

    def __enter__(self):
        def boom(*_):
            raise AssertionError("operation did not finish promptly")

        self.old = signal.signal(signal.SIGALRM, boom)
        signal.alarm(self.sec)

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old)
        return False


# ----------------------------- defaults / layering -------------------------
def test_minimal_valid_config_defaults(tmp_path):
    s = load(tmp_path, base())
    assert isinstance(s, Settings)
    assert s.db_path == ":memory:"
    assert s.max_body_bytes == 1_048_576
    assert s.idempotency_ttl_s == 3600
    assert s.env_name == "dev"
    assert s.audit_secret.get_secret_value() == SECRET
    assert s.tokens[TOK1].name == "alice"
    assert s.tokens[TOK1].role == "admin"


def test_env_overrides_file_and_defaults(tmp_path):
    f = write(tmp_path, base(max_body_bytes=2048, env_name="staging"))
    env = {
        "MLOPS_MAX_BODY_BYTES": "4096",
        "MLOPS_ENV_NAME": "prod",
        "MLOPS_DB_PATH": str(tmp_path / "x.db"),
    }
    s = load_settings(f, env=env)
    assert s.max_body_bytes == 4096  # env beats file
    assert s.env_name == "prod"  # env beats file
    assert s.db_path == str(tmp_path / "x.db")  # env beats default
    assert s.idempotency_ttl_s == 3600  # untouched default
    # file beats default when env is silent
    s2 = load_settings(f, env={})
    assert s2.max_body_bytes == 2048 and s2.env_name == "staging"


def test_json_encoded_env_tokens_and_secret_without_file():
    env = {
        "MLOPS_AUDIT_SECRET": SECRET,
        "MLOPS_TOKENS": json.dumps({"envtoken-12345": {"name": "a", "role": "viewer"}}),
    }
    s = load_settings(None, env=env)
    assert s.tokens["envtoken-12345"].name == "a"
    assert s.tokens["envtoken-12345"].role == "viewer"


def test_explicit_env_replaces_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("MLOPS_ENV_NAME", "prod")
    monkeypatch.setenv("MLOPS_MAX_BODY_BYTES", "999")
    s = load(tmp_path, base(), env={})
    assert s.env_name == "dev"
    assert s.max_body_bytes == 1_048_576
    # and env=None means os.environ
    s2 = load_settings(write(tmp_path, base()), env=None)
    assert s2.env_name == "prod" and s2.max_body_bytes == 999


# ----------------------------- validation table ----------------------------
BAD_CFGS = [
    ("secret_short", base(audit_secret="x" * 15)),
    ("secret_missing", {"tokens": {TOK1: {"name": "a", "role": "admin"}}}),
    ("role_root", base(tokens={TOK1: {"name": "a", "role": "root"}})),
    ("token_missing_name", base(tokens={TOK1: {"role": "admin"}})),
    ("body_zero", base(max_body_bytes=0)),
    ("body_negative", base(max_body_bytes=-1)),
    ("body_bool", base(max_body_bytes=True)),
    ("body_str", base(max_body_bytes="big")),
    ("body_too_big", base(max_body_bytes=64 * MIB + 1)),
    ("ttl_zero", base(idempotency_ttl_s=0)),
    ("ttl_negative", base(idempotency_ttl_s=-5)),
    ("ttl_bool", base(idempotency_ttl_s=True)),
    ("env_qa", base(env_name="qa")),
    ("unknown_key", base(surprise=1)),
    ("tokens_empty", base(tokens={})),
    ("token_short", base(tokens={"short": {"name": "a", "role": "admin"}})),
    ("token_space", base(tokens={"has space 123": {"name": "a", "role": "admin"}})),
    ("token_control", base(tokens={"tab\tinside-tok": {"name": "a", "role": "admin"}})),
    ("token_newline", base(tokens={"newline-tok\n1234": {"name": "a", "role": "admin"}})),
]


@pytest.mark.parametrize("cfg", [c for _, c in BAD_CFGS], ids=[i for i, _ in BAD_CFGS])
def test_invalid_config_raises_validation_failed(tmp_path, cfg):
    with pytest.raises(ValidationFailed):
        load(tmp_path, cfg)


def test_boundary_values_accepted(tmp_path):
    s = load(tmp_path, base(audit_secret="x" * 16, max_body_bytes=64 * MIB,
                            idempotency_ttl_s=1,
                            tokens={"12345678": {"name": "n", "role": "operator"}}))
    assert s.max_body_bytes == 64 * MIB and s.idempotency_ttl_s == 1
    assert load(tmp_path, base(max_body_bytes=1)).max_body_bytes == 1


def test_bad_env_values_raise_validation_failed(tmp_path):
    f = write(tmp_path, base())
    for env in ({"MLOPS_MAX_BODY_BYTES": "big"}, {"MLOPS_ENV_NAME": "qa"},
                {"MLOPS_TOKENS": "{not json"}, {"MLOPS_MAX_BODY_BYTES": "0"}):
        with pytest.raises(ValidationFailed):
            load_settings(f, env=env)


def test_bad_files_raise_validation_failed(tmp_path):
    with pytest.raises(ValidationFailed):  # invalid JSON
        load(tmp_path, "{not json")
    with pytest.raises(ValidationFailed):  # JSON array
        load(tmp_path, "[1, 2, 3]")
    with pytest.raises(ValidationFailed):  # missing file
        load_settings(str(tmp_path / "nope.json"), env={})
    pad = "x" * (MIB + 10)
    with pytest.raises(ValidationFailed):  # oversized (> 1 MiB), otherwise valid JSON
        load(tmp_path, json.dumps(base(db_path=pad)))


# ----------------------------- secrecy -------------------------------------
def test_secrets_not_in_repr_str_or_dump(tmp_path):
    s = load(tmp_path, base(tokens={TOK1: {"name": "alice", "role": "admin"},
                                    TOK2: {"name": "bob", "role": "viewer"}}))
    for text in (repr(s), str(s), s.model_dump_json()):
        assert SECRET not in text
        assert TOK1 not in text and TOK2 not in text
    assert s.audit_secret.get_secret_value() == SECRET


# ----------------------------- config_hash ---------------------------------
def test_config_hash_stable_and_prefixed(tmp_path):
    a = base(tokens={TOK1: {"name": "alice", "role": "admin"},
                     TOK2: {"name": "bob", "role": "viewer"}}, env_name="staging")
    b = {"env_name": "staging",
         "tokens": {TOK2: {"role": "viewer", "name": "bob"},
                    TOK1: {"role": "admin", "name": "alice"}},
         "audit_secret": SECRET}
    ha = config_hash(load(tmp_path, a))
    hb = config_hash(load(tmp_path, b))
    assert ha == hb
    assert ha.startswith("cfg_")
    assert config_hash(load(tmp_path, a)) == ha


def test_config_hash_changes_on_nonsecret_fields(tmp_path):
    ref = config_hash(load(tmp_path, base()))
    variants = [
        base(env_name="prod"),
        base(max_body_bytes=2048),
        base(tokens={TOK1: {"name": "alice", "role": "viewer"}}),  # role
        base(tokens={TOK1: {"name": "alicia", "role": "admin"}}),  # name
    ]
    hashes = {config_hash(load(tmp_path, v)) for v in variants}
    assert ref not in hashes
    assert len(hashes) == len(variants)


def test_config_hash_ignores_token_string_and_secret(tmp_path):
    ref = config_hash(load(tmp_path, base()))
    other_tok = base(tokens={"completely-different-tok": {"name": "alice", "role": "admin"}})
    other_secret = base(audit_secret="another-secret-0123456789")
    assert config_hash(load(tmp_path, other_tok)) == ref
    assert config_hash(load(tmp_path, other_secret)) == ref
    for tok in (TOK1, "completely-different-tok", SECRET):
        assert tok not in ref


# ----------------------------- immutability / bounds -----------------------
def test_settings_immutable_after_load(tmp_path):
    s = load(tmp_path, base())
    with pytest.raises(Exception):
        s.env_name = "prod"
    with pytest.raises(Exception):
        s.max_body_bytes = 5
    assert s.env_name == "dev" and s.max_body_bytes == 1_048_576


def test_many_tokens_bounded(tmp_path):
    toks = {f"token-{i:08d}": {"name": f"n{i}", "role": "viewer"} for i in range(10_000)}
    with alarm(5):
        with pytest.raises(ValidationFailed):
            load(tmp_path, base(tokens=toks))


def test_deeply_nested_json_bounded(tmp_path):
    nested = {}
    cur = nested
    for _ in range(100):
        cur["a"] = {}
        cur = cur["a"]
    with alarm(5):
        with pytest.raises(ValidationFailed):
            load(tmp_path, base(db_path=nested))
        with pytest.raises(ValidationFailed):
            load(tmp_path, "[" * 100 + "]" * 100)
