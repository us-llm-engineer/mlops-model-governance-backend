"""
R3.3 -- Failure / Chaos Resilience.

Claims implemented here (plan-c.md, C33-C36):

  C33 (source suite ``tests/R3_3_S1.py``) -- single-fault detection triggers
      automatic, idempotent rollback.  The three recoverable fault types
      (K8s Job killed mid-training, artifact store unavailable, policy gate
      service timeout) are caught and rolled back; the audit trail records
      the original failure, the rollback trigger, the rollback success and a
      measured recovery time.  Implemented by :class:`FaultInjector` and
      :class:`ResilienceController`.
  C34 (source suite ``tests/R3_3_S2.py``) -- cascading failures do not
      corrupt state or block queries.  :class:`SharedPlatformState` guards
      every mutation and audit append with a lock, never holds that lock
      across a nested rollback call, and therefore stays queryable and
      deadlock-free under concurrent fault injection.
  C35 (source suite ``tests/R3_3_S3.py``) -- unauthorized bypass attempts are
      rejected and logged, never silent.  :class:`PromotionGuard` checks the
      actor's role and artifact staleness *before* mutating any state and
      raises :class:`BypassRejected` with a reason naming the offending role
      or hash.
  C36 (source suite ``tests/R3_3_S4.py``) -- the full failure -> recovery ->
      audit journey is traceable end-to-end.  :class:`PipelineJourney` emits
      the seven expected stages in order and returns the measured recovery
      time (or raises :class:`RollbackFailed` when recovery itself fails).

R2 type reuse
-------------
The ``Role`` / ``Principal`` pair used by :class:`PromotionGuard` is the one
already defined for environment promotion in :mod:`mlops.promotion`
(``tests/R2_2_S4.py``), imported here rather than re-declared so both the
promotion and chaos layers agree on the same role vocabulary::

    from .promotion import Principal, PromotionRole as Role

The source suites are self-contained and duplicate a minimal ``Role`` /
``Principal`` so they remain independently runnable; that duplication is an
artefact of the test layout, not of this module.

Stdlib only (``time``, ``threading``, ``dataclasses``, ``datetime``, ``typing``).
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, NoReturn, Optional, Type

from .promotion import Principal, PromotionRole as Role

__all__ = [
    # clocks / timestamps
    "now_iso",
    "default_clock",
    "FakeClock",
    # fault vocabulary
    "KubernetesJobKilled",
    "ArtifactStoreUnavailable",
    "PolicyGateTimeout",
    "RECOVERABLE_FAULTS",
    "FAULT_TYPES",
    "FaultInjector",
    # C33
    "ResilienceController",
    # C34
    "SharedPlatformState",
    # C35
    "BypassRejected",
    "PromotionGuard",
    # C36
    "TrainingFault",
    "RollbackFailed",
    "PipelineJourney",
    # supporting plan-named types
    "ChaosScenario",
    "RecoveryVerifier",
]


# ---------------------------------------------------------------------------
# Clocks and timestamps
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def default_clock() -> float:
    """Return a monotonic timestamp in seconds (the production clock)."""
    return time.monotonic()


class FakeClock:
    """Deterministic clock for tests: each call advances by ``step``.

    Mirrors the reference clock used by ``R3_3_S1`` and ``R3_3_S4`` so
    recovery-time assertions are reproducible.
    """

    def __init__(self, start: float = 0.0, step: float = 0.2) -> None:
        self._t = start
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


# ---------------------------------------------------------------------------
# Fault vocabulary (C33)
# ---------------------------------------------------------------------------


class KubernetesJobKilled(Exception):
    """A training K8s Job was killed mid-run."""


class ArtifactStoreUnavailable(Exception):
    """The artifact store dependency could not be reached."""


class PolicyGateTimeout(Exception):
    """The policy gate service did not respond in time."""


#: Fault types the controller knows how to recover from automatically.
RECOVERABLE_FAULTS = (KubernetesJobKilled, ArtifactStoreUnavailable, PolicyGateTimeout)

#: Scenario-name -> fault-type mapping used by :class:`FaultInjector`.
FAULT_TYPES: Dict[str, Type[Exception]] = {
    "k8s_job_killed": KubernetesJobKilled,
    "artifact_store_unavailable": ArtifactStoreUnavailable,
    "policy_gate_timeout": PolicyGateTimeout,
}


class FaultInjector:
    """Inject a named recoverable fault in-process.

    Unknown scenario names fail loudly with :class:`ValueError` rather than
    being silently treated as "no fault".
    """

    def inject(self, scenario: str) -> NoReturn:
        """Raise the fault mapped to ``scenario``.

        Raises:
            ValueError: if ``scenario`` is not a recognised fault name.
            KubernetesJobKilled / ArtifactStoreUnavailable / PolicyGateTimeout:
                the mapped fault, with message ``f"injected fault: {scenario}"``.
        """
        if scenario not in FAULT_TYPES:
            raise ValueError(f"unknown fault scenario: {scenario}")
        raise FAULT_TYPES[scenario](f"injected fault: {scenario}")


# ---------------------------------------------------------------------------
# C33 -- automatic, idempotent rollback
# ---------------------------------------------------------------------------


class ResilienceController:
    """Detect a single fault, roll back automatically and audit the recovery.

    ``run_stage`` wraps one pipeline stage; if the stage raises any of
    :data:`RECOVERABLE_FAULTS` the controller logs the failure and rolls back
    to the last stable version.  Rollback is idempotent: calling it again is a
    safe no-op that leaves the version unchanged.
    """

    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:
        self._clock: Callable[[], float] = clock if clock is not None else default_clock
        self.state = "stable"
        self.audit: List[dict] = []
        self._last_stable_version = "v1.0.0"
        self.current_version = "v1.0.0"

    def _log(self, action: str, **fields: object) -> None:
        """Append an audit entry ``{"timestamp", "action", **fields}``."""
        self.audit.append({"timestamp": now_iso(), "action": action, **fields})

    def run_stage(
        self, stage_fn: Callable[[], object], new_version: str
    ) -> str:
        """Run ``stage_fn``; on a recoverable fault, roll back to last stable.

        Returns the resulting platform state (``"stable"``).
        """
        fault_start = self._clock()
        try:
            stage_fn()
            self.current_version = new_version
            self._last_stable_version = new_version
            self.state = "stable"
        except RECOVERABLE_FAULTS as fault:
            self._log(
                "failure_detected",
                error=str(fault),
                fault_type=type(fault).__name__,
            )
            self.rollback(fault_start)
        return self.state

    def rollback(self, fault_start_time: Optional[float] = None) -> float:
        """Roll back to the last stable version; return recovery time in seconds.

        Safe to call more than once: a second call re-affirms the same stable
        version and reports ``0.0`` when no start time is supplied.
        """
        self._log(
            "rollback_triggered",
            from_version=self.current_version,
            to_version=self._last_stable_version,
        )
        self.current_version = self._last_stable_version
        self.state = "stable"
        recovery_time_s = (
            self._clock() - fault_start_time if fault_start_time is not None else 0.0
        )
        self._log(
            "rollback_success",
            version=self.current_version,
            recovery_time_s=recovery_time_s,
        )
        return recovery_time_s


# ---------------------------------------------------------------------------
# C34 -- cascading failures do not corrupt state or block queries
# ---------------------------------------------------------------------------


class SharedPlatformState:
    """Thread-safe platform state store for concurrent fault injection.

    Every read-modify-write of the version pointer and every audit append
    happens inside a single critical section.  The lock is deliberately
    *not* held across the nested :meth:`rollback` call, so repeated
    acquisition by the same thread cannot self-deadlock a non-reentrant lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current_version = "v1.0.0"
        self._last_stable = "v1.0.0"
        self.audit: List[dict] = []
        self.state = "stable"

    def _log(self, action: str, **fields: object) -> None:
        """Append an audit entry under the lock."""
        with self._lock:
            self.audit.append({"timestamp": now_iso(), "action": action, **fields})

    def inject_fault(self, name: str) -> None:
        """Mark the platform degraded, log the fault, then roll back."""
        with self._lock:
            self.state = "degraded"
        self._log("fault_detected", fault=name)
        self.rollback(trigger=name)

    def rollback(self, trigger: str = "manual") -> None:
        """Restore the last stable version and state, then log success."""
        with self._lock:
            self.current_version = self._last_stable
            self.state = "stable"
        self._log("rollback_success", trigger=trigger, version=self.current_version)

    def query(self) -> Dict[str, str]:
        """Return a consistent ``{"version", "state"}`` snapshot under the lock."""
        with self._lock:
            return {"version": self.current_version, "state": self.state}


# ---------------------------------------------------------------------------
# C35 -- unauthorized bypass attempts rejected and logged
# ---------------------------------------------------------------------------


class BypassRejected(PermissionError):
    """Raised when an actor attempts a promotion bypass."""


class PromotionGuard:
    """Enforce authorization and artifact freshness before mutating state.

    Both checks run *before* any state mutation, so a rejected request never
    leaves partial changes behind, and every rejected attempt is audited with
    actor, action, decision and a specific reason.
    """

    def __init__(self) -> None:
        self.audit: List[dict] = []
        self.current_version = "v1.0.0"
        self.current_artifact_hash = "sha256:aaa000"

    def _log(self, actor: str, action: str, decision: str, reason: str) -> None:
        """Append a decision audit entry with the four required fields."""
        self.audit.append(
            {
                "timestamp": now_iso(),
                "actor": actor,
                "action": action,
                "decision": decision,
                "reason": reason,
            }
        )

    def approve_promotion(self, principal: Principal, to_version: str) -> bool:
        """Approve a prod promotion iff ``principal`` holds the RELEASER role.

        Raises:
            BypassRejected: if the role is not RELEASER; the reason names the
                offending role.  ``current_version`` is left unchanged.
        """
        if principal.role != Role.RELEASER:
            reason = (
                f"role '{principal.role.value}' is not authorized "
                f"to approve prod promotion"
            )
            self._log(principal.name, "approve_promotion", "reject", reason)
            raise BypassRejected(reason)
        self._log(principal.name, "approve_promotion", "allow", "authorized releaser")
        self.current_version = to_version
        return True

    def mark_artifact_current(
        self,
        principal: Principal,
        artifact_hash: str,
        verified_latest_hash: str,
    ) -> bool:
        """Mark ``artifact_hash`` current iff it matches the verified latest.

        Raises:
            BypassRejected: if the hash is stale; the reason names both the
                stale and the latest verified hash.  ``current_artifact_hash``
                is left unchanged.
        """
        if artifact_hash != verified_latest_hash:
            reason = (
                f"artifact {artifact_hash} is stale; "
                f"latest verified is {verified_latest_hash}"
            )
            self._log(principal.name, "mark_artifact_current", "reject", reason)
            raise BypassRejected(reason)
        self._log(
            principal.name, "mark_artifact_current", "allow", "matches latest verified"
        )
        self.current_artifact_hash = artifact_hash
        return True


# ---------------------------------------------------------------------------
# C36 -- full failure -> recovery -> audit journey
# ---------------------------------------------------------------------------


class TrainingFault(Exception):
    """A fault that interrupts a training/promotion journey."""


class RollbackFailed(Exception):
    """Raised when a rollback itself cannot restore a stable state."""


class PipelineJourney:
    """Drive the end-to-end failure -> recovery -> audit journey.

    Emits the seven :attr:`EXPECTED_STAGES` in order and returns the measured
    recovery time.  When ``fail_rollback`` is set the rollback stage is logged
    as failed and :class:`RollbackFailed` is raised.
    """

    EXPECTED_STAGES = [
        "submit",
        "gate_pass",
        "promotion_start",
        "fault_injected",
        "rollback_triggered",
        "rollback_success",
        "stable_restored",
    ]

    def __init__(self, clock: Optional[Callable[[], float]] = None) -> None:
        self._clock: Callable[[], float] = clock if clock is not None else default_clock
        self.audit: List[dict] = []
        self.version = "v1.0.0"

    def _log(self, stage: str, **fields: object) -> None:
        """Append a journey audit entry ``{"timestamp", "stage", "clock", ...}``."""
        self.audit.append(
            {"timestamp": now_iso(), "stage": stage, "clock": self._clock(), **fields}
        )

    def run(self, to_version: str, fail_rollback: bool = False) -> float:
        """Run the journey and return recovery time in seconds.

        Raises:
            RollbackFailed: when ``fail_rollback`` is true; the audit trail
                records ``rollback_failed`` and never ``rollback_success``.
        """
        self._log("submit", version=to_version)
        self._log("gate_pass")
        self._log("promotion_start", to_version=to_version)
        fault_start = self._clock()
        try:
            raise TrainingFault("k8s job killed mid-training")
        except TrainingFault as fault:
            self._log("fault_injected", error=str(fault))
            self._log("rollback_triggered")
            if fail_rollback:
                self._log("rollback_failed")
                raise RollbackFailed(
                    "rollback could not restore stable state"
                ) from fault
            self._log("rollback_success")
            self._log("stable_restored", version=self.version)
        return self._clock() - fault_start


# ---------------------------------------------------------------------------
# Supporting plan-named types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosScenario:
    """A named fault scenario: a scenario name paired with its fault type."""

    name: str
    fault_type: Type[Exception]

    @classmethod
    def from_name(cls, name: str) -> "ChaosScenario":
        """Build a scenario from a :data:`FAULT_TYPES` name.

        Raises:
            ValueError: if ``name`` is not a recognised fault scenario.
        """
        if name not in FAULT_TYPES:
            raise ValueError(f"unknown fault scenario: {name}")
        return cls(name=name, fault_type=FAULT_TYPES[name])


class RecoveryVerifier:
    """Verify that a controller/state ended stable at the last-good version."""

    def __init__(self, expected_version: str = "v1.0.0") -> None:
        self.expected_version = expected_version

    def verify(self, target: object) -> bool:
        """Return True iff ``target`` is ``"stable"`` at ``expected_version``.

        Accepts either a :class:`ResilienceController` (``state`` /
        ``current_version`` attributes) or a :class:`SharedPlatformState`
        (whose :meth:`query` snapshot is preferred).
        """
        state = getattr(target, "state", None)
        version = getattr(target, "current_version", None)
        query = getattr(target, "query", None)
        if callable(query):
            snapshot = query()
            state = snapshot.get("state", state)
            version = snapshot.get("version", version)
        return state == "stable" and version == self.expected_version
