"""
X3.3 -- Tenant reservations, sojourn/stability bounds and budget alerts (C69-C72).

This module lifts the X3.3 claim surface into one importable module:

* C69 ``TenantReservation`` -- a per-namespace reservation expressed as a
  tumbling-window credit budget. It is a *composition* wrapper: it programmes
  an :class:`~mlops.rbac.RBACEngine` quota (``max_concurrent_jobs``) for its
  namespace and delegates every allow/deny bookkeeping decision to
  ``RBACEngine.check_quota``. The engine only commits the increment on allow,
  so a rejected request leaves the delegated counter untouched.
* C70 ``sojourn_bound`` / ``stability`` -- the closed-form assured-sojourn
  bound and the admission-load ratio.
* C71 ``burn_rate`` -- error-budget consumption relative to the target error
  budget.
* C72 ``BudgetAlertTracker`` -- per-tenant consecutive-breach alerting and an
  independent cost-gate alert channel.

Standard library only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Optional

from .rbac import Principal, Role

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .audit import ChainedAuditStore
    from .gates import GateDecision
    from .rbac import RBACEngine

__all__ = [
    "StabilityError",
    "TenantReservation",
    "sojourn_bound",
    "stability",
    "burn_rate",
    "BudgetAlertTracker",
]

# Import StabilityError and validators from floor_guard
from . import floor_guard
StabilityError = floor_guard.StabilityError
_real = floor_guard._real
_count = floor_guard._count


class TenantReservation:
    """A tenant's per-window credit reservation over a shared RBAC engine.

    The reservation is enforced through the existing quota mechanism rather
    than by reimplementing admission logic: the constructor programmes
    ``rbac.quotas[namespace]["max_concurrent_jobs"]`` and :meth:`admit` calls
    ``rbac.check_quota``. ``RBACEngine`` commits a counter increment only when
    the request is allowed, so an over-budget request is rejected without
    mutating ``rbac.current_jobs``.
    """

    def __init__(
        self,
        rbac: "RBACEngine",
        namespace: str,
        credits_per_window: int,
        window_seconds: int,
    ) -> None:
        # Validate credits_per_window: must be int >= 1
        credits_per_window = _count(credits_per_window, "credits_per_window", lo=1)
        # Validate window_seconds: must be int > 0
        window_seconds = _count(window_seconds, "window_seconds", lo=1)

        self.rbac = rbac
        self.namespace = namespace
        self.credits_per_window = credits_per_window
        self.window_seconds = window_seconds
        self.rbac.quotas[namespace] = {
            "max_concurrent_jobs": credits_per_window,
        }
        self._window: Optional[int] = None

    def admit(self, now_ts: int, num_requests: int) -> bool:
        """Admit ``num_requests`` at ``now_ts``, refilling on a new window.

        The tumbling window index is ``now_ts // window_seconds``. Entering a
        new window resets the namespace's consumed credit counter to zero (no
        carryover, never negative) before the delegated quota check runs.
        Raises ValueError if time goes backwards.
        """
        # Validate inputs
        now_ts = _real(now_ts, "now_ts")
        num_requests = _count(num_requests, "num_requests", lo=0)

        window = int(now_ts // self.window_seconds)

        # Check time backwards
        if self._window is not None and window < self._window:
            raise ValueError("time went backwards")

        if window != self._window:
            self._window = window
            self.rbac.current_jobs[self.namespace] = 0

        principal = Principal(
            name=self.namespace or "reservation",
            role=Role.DEPLOYER,
            scope=self.namespace,
        )
        return self.rbac.check_quota(principal, self.namespace, int(num_requests))


def sojourn_bound(s_max: float, B: int, C: int) -> float:
    """Assured sojourn bound ``s_max * (1 + ceil(B / C))``.

    ``B`` is the window credit budget and ``C`` the per-interval capacity; the
    ceiling reflects whole service intervals only.
    Raises ValueError for non-finite or invalid inputs.
    """
    # Validate s_max: must be real finite number >= 0
    s_max = _real(s_max, "s_max")
    if s_max < 0:
        raise ValueError(f"s_max must be >= 0, got {s_max}")

    # Validate B: must be int >= 0
    B = _count(B, "B", lo=0)

    # Validate C: must be int > 0
    C = _count(C, "C", lo=1)

    return s_max * (1 + math.ceil(B / C))


def stability(B: int, s_max: float, C: int, window_seconds: int, strict: bool = False) -> dict:
    """Admission-load ratio and its inclusive stability verdict.

    ``rho_a = (B * s_max) / (C * window_seconds)``; the system is stable when
    ``rho_a <= 1`` (the boundary ``rho_a == 1`` counts as stable).

    With strict=True, raises StabilityError if rho_a > 1.
    """
    # Validate inputs
    B = _count(B, "B", lo=0)  # B >= 0
    s_max = _real(s_max, "s_max")
    if s_max < 0:
        raise ValueError(f"s_max must be >= 0, got {s_max}")
    C = _count(C, "C", lo=1)  # C >= 1
    window_seconds = _real(window_seconds, "window_seconds")
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}")

    rho_a = (B * s_max) / (C * window_seconds)
    stable = rho_a <= 1

    if strict and not stable:
        raise StabilityError("Stability constraint violated: rho_a > 1")

    return {"rho_a": rho_a, "stable": stable}


def burn_rate(observed_success_rate: float, target_success_rate: float) -> float:
    """Error-budget burn: observed error spend over the target error budget.

    ``(1 - observed) / (1 - target)``. A target with no error budget
    (``target_success_rate >= 1``) is invalid and raises ``ValueError`` rather
    than producing an infinite burn.
    Rejects NaN, bool, and non-finite values.
    """
    # Validate observed_success_rate
    observed = _real(observed_success_rate, "observed_success_rate")
    if not (0 <= observed <= 1):
        raise ValueError(f"observed_success_rate must be in [0, 1], got {observed}")

    # Validate target_success_rate
    target = _real(target_success_rate, "target_success_rate")
    if not (0 <= target < 1):
        raise ValueError(f"target_success_rate must be in [0, 1), got {target}")

    return (1 - observed) / (1 - target)


class BudgetAlertTracker:
    """Per-tenant consecutive-window SLO burn alerting plus cost alerts.

    An alert activates only after ``consecutive_windows`` consecutive windows
    whose burn is *strictly* above ``threshold``. Once active it stays active
    (``record_window`` keeps returning ``True``) but never re-appends an audit
    entry until a recovery window clears it. ``record_cost_alert`` is a
    separate channel and is unaffected by the burn-rate state.
    """

    def __init__(
        self,
        audit: "ChainedAuditStore",
        threshold: float,
        consecutive_windows: int,
    ) -> None:
        # Validate threshold: must be finite number >= 0
        threshold = _real(threshold, "threshold")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        # Validate consecutive_windows: must be int > 0
        consecutive_windows = _count(consecutive_windows, "consecutive_windows", lo=1)

        self.audit = audit
        self.threshold = threshold
        self.consecutive_windows = consecutive_windows
        self._tenants: Dict[str, Dict[str, Any]] = {}

    def _state_for(self, tenant: str) -> Dict[str, Any]:
        return self._tenants.setdefault(
            tenant, {"consecutive": 0, "flagged": False}
        )

    def record_window(self, tenant: str, burn: float) -> bool:
        """Record one window's burn, returning whether the alert is active.

        ``True`` means this call either newly raised the alert or kept an
        already-active alert active; ``False`` means no alert is active after
        this window.
        """
        state = self._state_for(tenant)

        if burn > self.threshold:
            state["consecutive"] += 1
            if state["flagged"]:
                return True
            if state["consecutive"] >= self.consecutive_windows:
                self.audit.append(
                    "system", "slo.budget_alert", tenant, "alert"
                )
                state["flagged"] = True
                return True
            return False

        state["consecutive"] = 0
        if state["flagged"]:
            self.audit.append(
                "system", "slo.budget_alert_cleared", tenant, "clear"
            )
            state["flagged"] = False
        return False

    def record_cost_alert(self, tenant: str, gate_decision: "GateDecision") -> None:
        """Audit a cost-budget alert when ``gate_decision`` did not pass.

        This is independent of any burn-rate alert state for the tenant.
        """
        if not gate_decision.passed:
            self.audit.append(
                "system", "cost.budget_alert", tenant, "alert"
            )
