"""Regression tests added by the independent F1 review (the review notes).

These are NEW tests (the six frozen F1 test files were not touched) that pin
three production-code defects found and fixed during the review:

1. mlops.svc.schemas.RuleModel params depth validation was off-by-one: a
   dict nested exactly 8 levels deep (the contract's own limit, "depth<=8")
   was wrongly rejected because the recursion charged an extra level of
   depth to scalar leaf values.
2. mlops.stats_scipy.slo_breach_pvalue did not enforce total >= 1 (the same
   "n>=1" trap the contract calls out for wilson_interval / scipy
   binomtest), so slo_breach_pvalue(0, 0, ...) silently returned nan instead
   of failing closed.
3. mlops.svc.extensions._IncidentCounts.open_counts() iterated the live
   IncidentManager._incidents dict directly with no lock. Under concurrent
   incident opens (e.g. a /metrics scrape running while another request
   opens an incident), this raised "RuntimeError: dictionary changed size
   during iteration", silently blacking out the mlops_incidents_open metric
   family exactly when incidents are most likely to be in flight.
"""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.incidents import IncidentManager
from mlops.kernel import SystemClock
from mlops.svc.extensions import _IncidentCounts
from mlops.svc.schemas import RuleModel
from mlops.stats_scipy import slo_breach_pvalue


def _nested(depth):
    d = {"v": 0}
    for _ in range(depth):
        d = {"nested": d}
    return d


def test_rule_model_params_depth_exactly_8_is_accepted():
    # Contract (api-contract.md section 4): "params: JSON-like, depth<=8".
    model = RuleModel(id="r", version=1, kind="k", params=_nested(8))
    assert model.params is not None


def test_rule_model_params_depth_9_is_rejected():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        RuleModel(id="r", version=1, kind="k", params=_nested(9))


def test_slo_breach_pvalue_zero_trials_raises_instead_of_nan():
    # Contract (api-contract.md section 1) calls out the same binomtest trap
    # documented for wilson_interval: n=0 must fail closed, not return nan.
    with pytest.raises(ValueError):
        slo_breach_pvalue(0, 0, 0.05)


def test_slo_breach_pvalue_negative_total_still_rejected():
    with pytest.raises(ValueError):
        slo_breach_pvalue(0, -1, 0.05)


def test_incident_counts_survive_concurrent_opens_no_dict_race():
    audit = ChainedAuditStore(b"review-race-secret-000000")
    manager = IncidentManager(audit, SystemClock())
    counts = _IncidentCounts(manager)

    errors = []
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            try:
                manager.open("alice", f"incident-{i}", {"pipeline_delay": True})
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(("writer", exc))
            i += 1

    def reader():
        while not stop.is_set():
            try:
                counts.open_counts()
            except Exception as exc:
                errors.append(("reader", exc))

    threads = [threading.Thread(target=writer) for _ in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join()

    assert errors == []
