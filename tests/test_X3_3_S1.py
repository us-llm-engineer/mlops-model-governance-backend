"""X3.3-S1: C69/C70 tenant reservations, sojourn bounds, and stability."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.rbac import RBACEngine
from mlops.slo_guard import TenantReservation, sojourn_bound, stability


def _reservation(namespace="tenant-a", credits=5, window=10, rbac=None):
    return TenantReservation(rbac or RBACEngine(), namespace, credits, window)


# X3.3-S1-01 / C69: the reservation admits exactly its window budget and a
# rejected over-budget request leaves the delegated quota state unchanged.
def test_reservation_admits_exact_window_budget_without_negative_credits():
    rbac = RBACEngine()
    reservation = _reservation(credits=5, rbac=rbac)

    assert reservation.admit(now_ts=3, num_requests=3) is True
    assert reservation.admit(now_ts=8, num_requests=2) is True
    used_before_rejection = rbac.current_jobs["tenant-a"]

    assert reservation.admit(now_ts=9, num_requests=1) is False
    assert used_before_rejection == 5
    assert rbac.current_jobs["tenant-a"] == 5


# X3.3-S1-02 / C69: a new tumbling window refills to B, rather than carrying
# an unused (or already used) balance from the preceding window.
def test_tumbling_window_boundary_refills_to_budget_without_carryover():
    rbac = RBACEngine()
    reservation = _reservation(credits=3, window=10, rbac=rbac)

    assert reservation.admit(now_ts=9, num_requests=1) is True
    assert reservation.admit(now_ts=10, num_requests=3) is True
    assert reservation.admit(now_ts=10, num_requests=1) is False
    assert rbac.current_jobs["tenant-a"] == 3


# X3.3-S1-03 / C69: splitting work into several calls cannot bypass a single
# window's reservation ceiling.
def test_multiple_admissions_share_one_window_credit_budget():
    rbac = RBACEngine()
    reservation = _reservation(credits=5, window=30, rbac=rbac)

    assert reservation.admit(now_ts=1, num_requests=2) is True
    assert reservation.admit(now_ts=20, num_requests=3) is True
    assert reservation.admit(now_ts=29, num_requests=1) is False
    assert rbac.current_jobs["tenant-a"] == 5


# X3.3-S1-04 / C69: tenants sharing one RBAC engine retain independent
# reservations; exhausting one namespace cannot consume another's credits.
def test_reservations_are_isolated_by_tenant_namespace():
    rbac = RBACEngine()
    alpha = _reservation(namespace="alpha", credits=2, rbac=rbac)
    beta = _reservation(namespace="beta", credits=2, rbac=rbac)

    assert alpha.admit(now_ts=1, num_requests=2) is True
    assert alpha.admit(now_ts=1, num_requests=1) is False
    assert beta.admit(now_ts=1, num_requests=2) is True
    assert rbac.current_jobs == {"alpha": 2, "beta": 2}


# X3.3-S1-05 / C70: non-divisible B/C uses ceil, not floor or ordinary
# division, in the published assured-sojourn bound.
def test_sojourn_bound_uses_ceiling_for_nondivisible_batch_size():
    assert sojourn_bound(s_max=1.25, B=5, C=2) == pytest.approx(5.0)


# X3.3-S1-06 / C70: a batch capacity larger than the reservation still has
# one service interval after the request's own service time.
def test_sojourn_bound_has_one_ceiling_batch_when_capacity_exceeds_budget():
    assert sojourn_bound(s_max=0.75, B=2, C=5) == pytest.approx(1.5)


# X3.3-S1-07 / C70: rho exactly one is stable (the contract's inclusive
# boundary) and the returned value is the stated admission-load ratio.
def test_stability_boundary_at_one_is_stable_with_exact_ratio():
    result = stability(B=10, s_max=2.0, C=2, window_seconds=10)

    assert result["rho_a"] == pytest.approx(1.0)
    assert result["stable"] is True


# X3.3-S1-08 / C70: any load strictly above one is flagged as unstable.
def test_stability_strictly_above_one_is_unstable():
    result = stability(B=11, s_max=2.0, C=2, window_seconds=10)

    assert result["rho_a"] == pytest.approx(1.1)
    assert result["stable"] is False
