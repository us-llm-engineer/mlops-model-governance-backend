"""Smoke test for the live-test harness itself: a real server subprocess boots, answers, and stops."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from live_tests.harness import History, LiveServer, git_commit


TOKENS = {"tk-admin-live-0000001": {"name": "root", "role": "admin"},
          "tk-view-live-00000001": {"name": "vic", "role": "viewer"}}


def test_server_starts_serves_and_stops(tmp_path):
    srv = LiveServer(tmp_path, tokens=TOKENS)
    srv.start(timeout_s=30)
    try:
        import httpx
        with httpx.Client(base_url=srv.base_url) as c:
            assert c.get("/healthz").json() == {"status": "ok", "env": "dev"}
    finally:
        srv.stop()
    assert srv._proc.poll() is not None


def test_kill9_actually_kills_the_process_group(tmp_path):
    srv = LiveServer(tmp_path, tokens=TOKENS)
    srv.start(timeout_s=30)
    pid = srv._proc.pid
    srv.kill9()
    assert srv._proc.poll() is not None
    with open("/proc/self/status") as _:
        pass
    assert not os.path.exists(f"/proc/{pid}")


def test_history_records_ok_denied_crash_and_writes_jsonl(tmp_path):
    h = History("unit-test", seed=1, out_dir=tmp_path)

    class Boom(Exception):
        pass

    def fail():
        raise Boom("x")

    result, outcome = h.try_step("s1", lambda: {"a": 1})
    assert outcome == "ok"
    result, outcome = h.try_step("s2", fail)
    assert outcome == "crash"
    lines = (tmp_path / "history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3  # run.start + s1 + s2
    assert json.loads(lines[-1])["outcome"] == "crash"
    s = h.summary()
    assert s["commit"] == git_commit() and s["by_outcome"]["crash"] == 1 and s["by_outcome"]["ok"] == 1


def test_try_step_returns_result_first_so_scenario_code_can_chain_it(tmp_path):
    """Every scenario driver destructures as `value, outcome = h.try_step(...)` to chain a real
    ID/dict into the next call. try_step's return order must be (result, outcome), not the
    reverse -- the reverse silently binds the outcome STRING where the caller expects the value,
    which crashes (or worse, runs) far from the actual defect. Caught for real in the S1 driver."""
    h = History("unit-test", seed=1, out_dir=tmp_path)
    version_id, outcome = h.try_step("dataset.register", lambda: "ds_abcdef0123456789")
    assert outcome == "ok"
    assert version_id == "ds_abcdef0123456789"
    assert isinstance(outcome, str) and version_id != outcome
