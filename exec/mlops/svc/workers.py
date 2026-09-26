"""F3.6 Background workers for periodic tasks (drift evaluation, incident escalation).

Provides PeriodicWorker for async-wrapped sync callables that run at fixed intervals,
drift_evaluation_tick and incident_escalation_tick pure functions for use in worker loops,
and WorkerRegistry for managing multiple workers via an anyio TaskGroup.
"""

import math
from collections import deque
from typing import Callable, Optional

import anyio

from mlops.kernel import ValidationFailed


class PeriodicWorker:
    """
    A worker that runs a synchronous function periodically.

    Runs the given function in a loop with a fixed interval, checking for a stop
    event between iterations. Exceptions are either passed to an optional on_error
    callback or recorded in a bounded error log (maxlen 100). The loop continues
    after exceptions; only the stop_event causes the worker to exit cleanly.
    """

    def __init__(
        self,
        name: str,
        interval_s: float,
        fn: Callable[[], None],
        *,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize a PeriodicWorker.

        Args:
            name: worker name (for identification)
            interval_s: interval in seconds (must be in (0, 3600], not bool, else ValidationFailed)
            fn: synchronous callable that takes no arguments
            on_error: optional callback for exceptions; if provided, exceptions are
                     passed to it and NOT recorded in self.errors

        Raises:
            ValidationFailed: if interval_s is not a float in (0, 3600], or is bool
        """
        # Validate interval_s: must be float in (0, 3600], not bool
        if isinstance(interval_s, bool):
            raise ValidationFailed("interval_s cannot be a boolean")
        if not isinstance(interval_s, (int, float)):
            raise ValidationFailed("interval_s must be numeric")
        interval_s = float(interval_s)
        if math.isnan(interval_s) or math.isinf(interval_s):
            raise ValidationFailed("interval_s cannot be NaN or Inf")
        if interval_s <= 0 or interval_s > 3600:
            raise ValidationFailed("interval_s must be in (0, 3600]")

        self.name = name
        self.interval_s = interval_s
        self.fn = fn
        self.on_error = on_error
        self.errors: deque = deque(maxlen=100)
        self._stop_event: Optional[anyio.Event] = None

    async def run(self, *, stop_event: "anyio.Event") -> None:
        """
        Run the worker loop until stop_event is set.

        Calls fn repeatedly with interval_s sleep between iterations. Checks
        stop_event BETWEEN iterations and cancels via anyio.move_on_after.
        Exceptions from fn are either passed to on_error or recorded in self.errors.
        The loop continues after exceptions and only exits when stop_event is set.

        Args:
            stop_event: anyio.Event to signal shutdown
        """
        self._stop_event = stop_event

        while not stop_event.is_set():
            # Run the sync function in a thread to avoid blocking the event loop
            try:
                await anyio.to_thread.run_sync(self.fn)
            except Exception as exc:
                if self.on_error is not None:
                    self.on_error(exc)
                else:
                    self.errors.append(exc)

            # Check stop event before sleeping; use move_on_after to allow
            # cancellation during the sleep
            if stop_event.is_set():
                break

            with anyio.fail_after(self.interval_s + 1, shield=True):
                try:
                    with anyio.move_on_after(self.interval_s):
                        await stop_event.wait()
                        # If we get here, stop_event was set during sleep
                        break
                except anyio.get_cancelled_exc_class():
                    # move_on_after hit its timeout, continue the loop
                    continue

    def stop(self) -> None:
        """Signal the worker to stop (for backward compatibility)."""
        if self._stop_event is not None:
            self._stop_event.set()


def drift_evaluation_tick(monitor_by_name: dict, baseline_repo, frame_reports: Optional[dict] = None) -> None:
    """
    Evaluate drift for all monitors and upsert baselines for non-ok levels.

    For each monitor in the dictionary:
    - Calls monitor.check()
    - If the result's level is not "ok", upserts the current window as the new baseline
    - If frame_reports is given and the monitor has >= 30 observations, also builds a
      richer pandas-based multi-window report (mlops.ext.drift_frame.DriftWindowFrame)
      and stores it in frame_reports[name]. Best-effort: a report failure never affects
      the baseline upsert above, which remains the tick's real contract.

    This is a pure function taking its dependencies as arguments, making it
    trivially unit-testable without a real worker loop.

    Args:
        monitor_by_name: dict mapping monitor names to DriftMonitor instances
        baseline_repo: repository with upsert_drift_baseline(name, reference) method
        frame_reports: optional dict this tick populates with a richer per-monitor
                       report (caller-owned, e.g. exposed by a status endpoint/demo)
    """
    for name, monitor in monitor_by_name.items():
        result = monitor.check()
        if result.level != "ok":
            # Use the monitor's private _observations attribute (sliding window).
            # A real DriftMonitor keeps this as a collections.deque; OpsRepo's real
            # validation requires an actual list (only the frozen tests' fake
            # monitor already stored a list, which is why this needed a real
            # OpsRepo to surface at all) -- convert explicitly.
            baseline_repo.upsert_drift_baseline(name, list(monitor._observations))

        if frame_reports is not None and len(monitor._observations) >= 30:
            try:
                from mlops.ext.drift_frame import DriftWindowFrame

                pairs = [(float(i), v) for i, v in enumerate(monitor._observations)]
                frame = DriftWindowFrame(pairs)
                psi_series = frame.rolling_psi(monitor._reference)
                frame_reports[name] = {
                    "daily_summary": frame.to_daily_summary().to_dict(orient="records"),
                    "rolling_psi_tail": [float(v) for v in psi_series.tail(5).tolist()],
                }
            except Exception:
                pass


def incident_escalation_tick(manager) -> int:
    """
    Escalate overdue incidents and report how many actually escalated.

    Iterates manager.list_open_incident_ids() and calls
    manager.escalate_if_overdue(incident_id) for each, per IncidentManager's
    real per-incident contract (escalate_if_overdue takes an incident_id and
    returns the resulting tier; the audit trail records only a tier rise).
    A rise between before/after tiers counts as one escalation this tick.

    Args:
        manager: IncidentManager instance (or duck-typed equivalent) exposing
                 list_open_incident_ids() -> list[str], get(incident_id) -> dict
                 with "current_tier", and escalate_if_overdue(incident_id) -> int

    Returns:
        int: number of incidents whose tier rose this tick
    """
    escalated = 0
    for incident_id in manager.list_open_incident_ids():
        before = manager.get(incident_id)["current_tier"]
        after = manager.escalate_if_overdue(incident_id)
        if after > before:
            escalated += 1
    return escalated


class WorkerRegistry:
    """
    Registry for managing multiple PeriodicWorker instances.

    Provides add(), start_all(), and stop_all() methods for lifecycle management.
    Designed to be started from a FastAPI lifespan context manager.
    """

    def __init__(self):
        """Initialize an empty worker registry."""
        self.workers: list[PeriodicWorker] = []

    def add(self, worker: PeriodicWorker) -> None:
        """
        Add a worker to the registry.

        Args:
            worker: PeriodicWorker instance to add
        """
        self.workers.append(worker)

    def start_all(self, task_group: "anyio.abc.TaskGroup") -> None:
        """
        Start all registered workers in the given task group.

        Each worker is spawned via task_group.start_soon with its own stop_event.

        Args:
            task_group: anyio TaskGroup to spawn workers in
        """
        for worker in self.workers:
            stop_event = anyio.Event()
            worker._stop_event = stop_event

            # Create a wrapper to pass stop_event (use default args to avoid late binding)
            async def run_with_stop(w=worker, se=stop_event):
                await w.run(stop_event=se)

            task_group.start_soon(run_with_stop)

    def stop_all(self) -> None:
        """
        Signal all workers to stop.

        Sets the stop event for each worker, allowing them to exit cleanly.
        """
        for worker in self.workers:
            if worker._stop_event is not None:
                worker._stop_event.set()
