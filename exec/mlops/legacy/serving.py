"""
Blue-green / canary serving and traffic management (Claims C21-C24).

Maps to requirement R2.3 (Kubernetes Serving & Traffic Management) and is
lifted from the frozen reference implementations in ``tests/R2_3_S2.py``
(C22: configurable canary phase), ``tests/R2_3_S3.py`` (C23: automatic
rollback under 5s with no downtime) and ``tests/R2_3_S4.py`` (C24: blue-green
two-Deployment setup behind a shared Service). K8s objects are local stubs --
no API calls, no service mesh.

UNIFICATIONS (deliberate)
-------------------------
Two incompatible variants are merged here:

* ``DeploymentState``: R2_3_S4 defines only ACTIVE/RETIRED; R2_3_S3 adds
  ROLLED_BACK for a failed canary. The union of all three is used.
* ``BlueGreenController``: R2_3_S3 adds ``rollback_canary``/ROLLED_BACK and a
  timestamped audit entry; R2_3_S4 adds ``build_service``/``SharedService``.
  Both ``rollback_canary``, ``promote_canary_to_stable`` and ``build_service``
  now live on one controller. R2_3_S4's un-timestamped promote entry gains the
  same ``timestamp`` field R2_3_S3 already had, which is additive and does not
  affect the S4 assertions.

Standard library only (time, dataclasses, enum, datetime). Serving decisions
never both roll back and promote on the same decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

from .metrics import MetricsCollector


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class DeploymentState(Enum):
    """Lifecycle state of a Deployment (union of R2_3_S3 and R2_3_S4)."""

    ACTIVE = "active"
    RETIRED = "retired"
    ROLLED_BACK = "rolled_back"


@dataclass
class Deployment:
    """A single deployment version receiving a share of traffic."""

    name: str
    role: str
    traffic_weight: float
    state: DeploymentState = DeploymentState.ACTIVE

    def weight_for_ingress(self) -> float:
        """Return the traffic weight used by the Ingress rule."""
        return self.traffic_weight


class ServiceConfigError(Exception):
    """Raised when a SharedService violates the blue-green invariants."""


@dataclass
class SharedService:
    """A shared Service routing to exactly two Deployments by weight."""

    name: str
    targets: dict

    def validate(self) -> bool:
        """Return True if the service is a valid two-Deployment blue-green split."""
        if len(self.targets) != 2:
            raise ServiceConfigError(
                "Blue-green service must route to exactly two Deployments, "
                f"got {len(self.targets)}"
            )
        total = sum(self.targets.values())
        if abs(total - 1.0) > 1e-6:
            raise ServiceConfigError(f"Traffic weights must sum to 1.0, got {total}")
        return True


class ModelServingDeployment:
    """Supporting type: a serving Deployment view exposing its Ingress weight."""

    def __init__(
        self,
        name: str,
        role: str,
        traffic_weight: float,
        state: DeploymentState = DeploymentState.ACTIVE,
    ) -> None:
        self.name = name
        self.role = role
        self.traffic_weight = traffic_weight
        self.state = state

    def weight_for_ingress(self) -> float:
        """Return the traffic weight to apply at the Ingress."""
        return self.traffic_weight


class BlueGreenController:
    """Owns the stable/canary Deployments and drives promote/rollback.

    Unified from R2_3_S3 (ROLLED_BACK rollback) and R2_3_S4 (shared Service
    setup).
    """

    def __init__(
        self,
        stable_name="model-stable",
        canary_name="model-canary",
        canary_weight=0.1,
    ):
        self.stable = Deployment(stable_name, "stable", 1.0 - canary_weight)
        self.canary = Deployment(canary_name, "canary", canary_weight)
        self.audit_log: list = []

    def build_service(self, name="model-service") -> SharedService:
        """Build the shared Service routing to both Deployments' weights."""
        return SharedService(
            name=name,
            targets={
                self.stable.name: self.stable.traffic_weight,
                self.canary.name: self.canary.traffic_weight,
            },
        )

    def rollback_canary(self, reason: str) -> dict:
        """Roll the canary back; stable stays ACTIVE and serves all traffic."""
        start = time.perf_counter()
        self.canary.traffic_weight = 0.0
        self.canary.state = DeploymentState.ROLLED_BACK
        # stable is never touched/stopped -- it remains ACTIVE throughout
        self.stable.traffic_weight = 1.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        entry = {
            "action": "rollback_canary",
            "reason": reason,
            "duration_ms": elapsed_ms,
            "stable_state": self.stable.state.value,
            "stable_weight": self.stable.traffic_weight,
            "timestamp": _now(),
        }
        self.audit_log.append(entry)
        return entry

    def promote_canary_to_stable(self) -> dict:
        """Swap the canary to stable at 100% and retire the old stable."""
        old_stable = self.stable
        self.canary.role = "stable"
        self.canary.traffic_weight = 1.0
        self.canary.state = DeploymentState.ACTIVE
        old_stable.state = DeploymentState.RETIRED
        old_stable.traffic_weight = 0.0
        self.stable, self.canary = self.canary, old_stable
        entry = {
            "action": "promote_canary",
            "new_stable": self.stable.name,
            "retired": self.canary.name,
            "timestamp": _now(),
        }
        self.audit_log.append(entry)
        return entry


class CanaryAnalyzer:
    """Runs a canary phase that stops at ``duration_s`` seconds or
    ``max_requests`` canary requests, whichever comes first. Both are
    explicit constructor parameters -- neither is hardcoded."""

    def __init__(self, duration_s: float = 300.0, max_requests: int = 1000):
        self.duration_s = duration_s
        self.max_requests = max_requests
        self.collector = MetricsCollector()
        self.phase_start: Optional[float] = None

    def start_phase(self, now: float) -> None:
        """Mark the start of the canary phase at time ``now``."""
        self.phase_start = now

    def record_request(self, track: str, latency_ms: float, now: float) -> None:
        """Record a request observation timestamped at ``now``."""
        self.collector.record(track, latency_ms, now)

    def phase_complete(self, now: float) -> bool:
        """Return True once the duration or the canary request cap is reached."""
        elapsed = now - self.phase_start
        n_canary = len(self.collector.latencies("canary"))
        return elapsed >= self.duration_s or n_canary >= self.max_requests

    def stop_reason(self, now: float) -> str:
        """Return why the phase stopped, or ``"in_progress"`` if not yet done."""
        elapsed = now - self.phase_start
        n_canary = len(self.collector.latencies("canary"))
        if n_canary >= self.max_requests:
            return "request_cap_reached"
        if elapsed >= self.duration_s:
            return "duration_elapsed"
        return "in_progress"


__all__ = [
    "DeploymentState",
    "Deployment",
    "ServiceConfigError",
    "SharedService",
    "ModelServingDeployment",
    "BlueGreenController",
    "CanaryAnalyzer",
]
