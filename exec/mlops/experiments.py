"""
X2.1 / X2.2 -- Experiment Tracking, Checkpoints, Backfill & Pivots.

An append-only, per-run experiment log layered over the X1 pipeline run state
machine, content-addressed artifact store and chained audit log. Metrics may be
logged concurrently without losing updates; only Running or Completed runs are
writable; checkpoints are content-addressed and keyed by (run_id, step);
backfilling replays a run's own params/checkpoints without ever re-running a
pipeline, and refuses to overwrite a live metric. ``pivot`` and ``best_run``
give latest-value-wins tabular views and max/min selection with earliest-start
tie-breaking.

Stdlib only.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from mlops.state_machine import RunState

__all__ = [
    "ExperimentError",
    "RunNotWritableError",
    "BackfillConflictError",
    "LogEntry",
    "ExperimentTracker",
]


def _now_epoch_int() -> int:
    """Return the current epoch time truncated to whole seconds."""
    return int(time.time())


class ExperimentError(Exception):
    """Base class for experiment-tracking errors."""


class RunNotWritableError(ExperimentError):
    """Raised when a metric is logged against a run that is not writable."""


class BackfillConflictError(ExperimentError):
    """Raised when a backfill would overwrite a live metric entry."""


@dataclass
class LogEntry:
    """A single append-only entry in a run's experiment log."""

    kind: str
    name: str
    value: Any
    step: Optional[int]
    backfilled: bool = False
    ts: int = field(default_factory=_now_epoch_int)


class ExperimentTracker:
    """Track params, metrics and checkpoints for pipeline runs."""

    WRITABLE_STATES = (RunState.RUNNING, RunState.COMPLETED)

    def __init__(self, run_manager, artifact_store, audit):
        self.run_manager = run_manager
        self.artifact_store = artifact_store
        self.audit = audit
        self._lock = threading.Lock()
        self._log: Dict[str, List[LogEntry]] = {}
        self._run_order: List[str] = []

    def _record_run(self, run_id: str) -> None:
        """Record a run's first-seen order and ensure it has a log bucket."""
        if run_id not in self._run_order:
            self._run_order.append(run_id)
        self._log.setdefault(run_id, [])

    def start_run(self, run_id: str, params: dict) -> None:
        """Submit and start a run, logging each param as a "param" entry."""
        with self._lock:
            self.run_manager.submit_run(run_id)
            self.run_manager.transition_to_running(run_id)
            self._record_run(run_id)
            for name, value in params.items():
                self._log[run_id].append(
                    LogEntry(
                        kind="param",
                        name=name,
                        value=value,
                        step=None,
                        ts=_now_epoch_int(),
                    )
                )
            self.audit.append("experiment", "experiment.start", run_id, "allow")

    def log_metric(
        self, run_id: str, name: str, value, step: Optional[int] = None
    ) -> None:
        """Append a metric entry; raise RunNotWritableError if not writable."""
        with self._lock:
            run = self.run_manager.get_run(run_id)
            if run is None or run.state not in self.WRITABLE_STATES:
                raise RunNotWritableError(
                    f"Run is not writable: {run_id}"
                )
            self._record_run(run_id)
            self._log[run_id].append(
                LogEntry(
                    kind="metric",
                    name=name,
                    value=value,
                    step=step,
                    backfilled=False,
                    ts=_now_epoch_int(),
                )
            )
            self.audit.append("experiment", "experiment.log_metric", run_id, "allow")

    def log_checkpoint(self, run_id: str, step: int, content: bytes) -> str:
        """Store a checkpoint artifact and append its entry; return the id."""
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            artifact_id = self.artifact_store.store_artifact(
                content,
                filename=f"{run_id}_step{step}",
                artifact_type="checkpoint",
                created_at=created_at,
            )
            self._record_run(run_id)
            self._log[run_id].append(
                LogEntry(
                    kind="checkpoint",
                    name="checkpoint",
                    value=artifact_id,
                    step=step,
                    ts=_now_epoch_int(),
                )
            )
            self.audit.append(
                "experiment", "experiment.log_checkpoint", run_id, "allow"
            )
            return artifact_id

    def get_checkpoint(self, run_id: str, step: int) -> bytes:
        """Return the checkpoint bytes stored for (run_id, step)."""
        for entry in reversed(self._log.get(run_id, [])):
            if entry.kind == "checkpoint" and entry.step == step:
                return self.artifact_store.retrieve_artifact(entry.value)
        raise KeyError(f"No checkpoint for {run_id} at step {step}")

    def backfill_metric(
        self, run_id: str, name: str, replay_fn: Callable[[dict], float]
    ) -> float:
        """Recompute a missing metric from the run's own params/checkpoints."""
        with self._lock:
            entries = self._log.get(run_id, [])
            for entry in entries:
                if (
                    entry.kind == "metric"
                    and entry.name == name
                    and not entry.backfilled
                ):
                    raise BackfillConflictError(
                        f"Metric already exists: {name} for {run_id}"
                    )

            context = {
                "params": {
                    e.name: e.value
                    for e in entries
                    if e.kind == "param"
                },
                "checkpoints": {
                    e.step: self.artifact_store.retrieve_artifact(e.value)
                    for e in entries
                    if e.kind == "checkpoint"
                },
            }

            result = replay_fn(context)
            self._record_run(run_id)
            self._log[run_id].append(
                LogEntry(
                    kind="backfill",
                    name=name,
                    value=result,
                    step=None,
                    backfilled=True,
                    ts=_now_epoch_int(),
                )
            )
            self.audit.append("experiment", "experiment.backfill", run_id, "allow")
            return result

    def entries(self, run_id: str) -> List[LogEntry]:
        """Return a shallow copy of a run's log in append order."""
        return list(self._log.get(run_id, []))

    def pivot(self, metric_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return {run_id: {metric: latest value}} for runs having a metric."""
        wanted = set(metric_names)
        table: Dict[str, Dict[str, Any]] = {}
        for run_id, log in self._log.items():
            run_metrics: Dict[str, Any] = {}
            for entry in log:
                if (
                    entry.kind in ("metric", "backfill")
                    and entry.name in wanted
                ):
                    run_metrics[entry.name] = entry.value
            if run_metrics:
                table[run_id] = run_metrics
        return table

    def best_run(self, metric: str, mode: str = "max") -> Optional[str]:
        """Return the run with the best latest value for a metric, or None."""
        table = self.pivot([metric])
        best_id = None
        best_val = None
        best_idx = None
        for run_id, metrics in table.items():
            if metric not in metrics:
                continue
            value = metrics[metric]
            idx = (
                self._run_order.index(run_id)
                if run_id in self._run_order
                else len(self._run_order)
            )
            better = False
            if best_id is None:
                better = True
            elif mode == "min":
                if value < best_val or (value == best_val and idx < best_idx):
                    better = True
            else:
                if value > best_val or (value == best_val and idx < best_idx):
                    better = True
            if better:
                best_id = run_id
                best_val = value
                best_idx = idx
        return best_id
