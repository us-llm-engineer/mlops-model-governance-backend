"""Live-test harness: real server subprocess, a history recorder, and component evidence checks.

Per the /live-test skill: drive only the public surface (HTTP/SDK/CLI); observation may read the
DB and audit export read-only. Every scenario carries {seed, commit} and a replayable history where
timeouts/unknowns are first-class outcomes, not failures.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

REPO = Path(__file__).resolve().parent.parent
EXEC = REPO / "exec"


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


@dataclass
class Event:
    """One recorded step. outcome is one of ok / denied / info / timeout / crash."""
    step: str
    outcome: str
    detail: dict
    t: float = field(default_factory=time.time)


class History:
    """Append-only run record. Never asserts by itself -- checks run over the recorded events afterward."""

    def __init__(self, scenario: str, seed: int, out_dir: Path):
        self.scenario, self.seed, self.commit = scenario, seed, git_commit()
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[Event] = []
        self._path = out_dir / "history.jsonl"
        self._path.write_text("")
        self.record("run.start", "info", {"scenario": scenario, "seed": seed, "commit": self.commit})

    def record(self, step: str, outcome: str, detail: Optional[dict] = None) -> Event:
        e = Event(step, outcome, detail or {})
        self.events.append(e)
        with open(self._path, "a") as f:
            f.write(json.dumps({"t": e.t, "step": e.step, "outcome": e.outcome, "detail": e.detail}, default=str) + "\n")
        return e

    def try_step(self, step: str, fn: Callable[[], Any], *, timeout_outcome: str = "timeout") -> tuple[Any, str]:
        """Run fn(); classify the outcome without raising, so a flow test never dies on a single step.

        Returns (value, outcome) -- value first, so scenario code can chain a real ID/dict into the
        next call by writing `value, outcome = h.try_step(...)`. On any non-"ok" outcome, value is
        None. Getting this order backwards silently binds the outcome STRING where a real value was
        expected (a real defect caught in the S1 driver: 'str' object has no attribute 'get').
        """
        try:
            result = fn()
            return result, self.record(step, "ok", {"result": _brief(result)}).outcome
        except httpx.TimeoutException as e:
            return None, self.record(step, timeout_outcome, {"error": str(e)}).outcome
        except MlopsApiErrorLike as e:
            outcome = "denied" if getattr(e, "status", 0) in (401, 403) else "info"
            return None, self.record(step, outcome, {"status": getattr(e, "status", None), "message": str(e)}).outcome
        except Exception as e:  # noqa: BLE001 - a live test records everything, including crashes
            return None, self.record(step, "crash", {"error": f"{type(e).__name__}: {e}"}).outcome

    def summary(self) -> dict:
        from collections import Counter
        return {"scenario": self.scenario, "seed": self.seed, "commit": self.commit,
                "events": len(self.events), "by_outcome": dict(Counter(e.outcome for e in self.events))}


def _brief(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _brief(x) for k, x in list(v.items())[:20]}
    if isinstance(v, list):
        return [_brief(x) for x in v[:5]] + (["...(%d more)" % (len(v) - 5)] if len(v) > 5 else [])
    if isinstance(v, str) and len(v) > 300:
        return v[:300] + "...(truncated)"
    return v


class MlopsApiErrorLike(Exception):
    """Duck-typed so harness.py does not import exec/mlops before sys.path is set by the caller."""


class LiveServer:
    """A real `python -m mlops.svc` subprocess on a free port, with a real SQLite file.

    Never used to advance a flow directly -- only to start/stop/kill the process being tested.
    All state changes go through HTTP/SDK/CLI.
    """

    def __init__(self, work_dir: Path, *, tokens: dict, extra_env: Optional[dict] = None, port: Optional[int] = None):
        self.work_dir = work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        self.port = port or free_port()
        self.db_path = str(work_dir / "mlops.db")
        self.log_path = work_dir / "server.log"
        env = {
            **os.environ,
            "MLOPS_AUDIT_SECRET": "live-test-secret-" + "0" * 20,
            "MLOPS_TOKENS": json.dumps(tokens),
            "MLOPS_DB_PATH": self.db_path,
            "MLOPS_METRICS_PUBLIC": "true",
            "PYTHONPATH": str(EXEC),
            **(extra_env or {}),
        }
        self.env = env
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_s: float = 60) -> None:
        log = open(self.log_path, "a")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "mlops.svc", "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=str(EXEC), env=self.env, stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        deadline = time.time() + timeout_s
        with httpx.Client(base_url=self.base_url, timeout=2.0) as c:
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"server exited early (code {self._proc.returncode}); see {self.log_path}")
                try:
                    if c.get("/healthz").status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                time.sleep(0.3)
        raise TimeoutError(f"server did not become healthy within {timeout_s}s; see {self.log_path}")

    def kill9(self) -> None:
        """Ungraceful kill (recovery flow): SIGKILL the process group, no shutdown hooks run."""
        if self._proc and self._proc.poll() is None:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            self._proc.wait(timeout=10)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
