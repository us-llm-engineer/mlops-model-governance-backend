"""
CI/CD policy gates, unified decision record and audit logging (Claims C13/C15/C16).

Maps to requirement R2.1 (CI/CD Policy Gates) and is lifted from the frozen
reference implementations in the Round-2 suites:

    C13  tests/R2_1_S1.py   DataContractGate -- reject malformed data pre-train
    C15  tests/R2_1_S3.py   PolicyEngine     -- declarative gate audit logging
    C16  tests/R2_1_S4.py   PolicyEngine     -- per-rule evaluation latency budget

Deliberate unification
----------------------
R2_1_S3 and R2_1_S4 each define their own, incompatible ``PolicyEngine`` and
``GateDecision`` variants. Here they are merged into a single pair:

* ``GateDecision`` carries both the declarative audit fields (``rule_name``,
  ``passed``, ``reason``, ``predicates``, ``timestamp``) and the latency fields
  (``latency_ms``, ``budget_violated``) with the latter defaulting to zero/false,
  so a gate built for the S1/S3 surface is still a valid S4 decision.
* ``AuditEntry`` is a dataclass that is simultaneously dict-like (via
  ``__getitem__``, satisfying the S3 declarative-audit view:
  ``rule_name``/``predicates``/``result``/``reason``/``timestamp``/
  ``approval_trace``) and attribute-accessible (satisfying the S4 latency view:
  ``latency_ms``/``budget_violated``). One record therefore serves both suites.
* ``PolicyEngine`` performs the S3 append-only structured audit *and* the S4
  wall-clock latency measurement in the same ``evaluate`` call.

Standard library only.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "GateDecision",
    "AuditEntry",
    "DataContractGate",
    "PolicyEngine",
    "now_iso",
]


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GateDecision:
    """A single gate's declarative decision plus its latency measurement.

    Field order is frozen: the S1/S3 declarative fields first, then the S4
    latency fields (which default so existing gates need not supply them).
    """

    rule_name: str
    passed: bool
    reason: str
    predicates: list
    timestamp: str
    latency_ms: float = 0.0
    budget_violated: bool = False


@dataclass
class AuditEntry:
    """An append-only audit record for one gate evaluation.

    Supports the declarative-audit view as a mapping (``entry["result"]``,
    ``entry["reason"]``, ...) and the latency view as attributes
    (``entry.latency_ms``, ``entry.budget_violated``), so a single entry
    satisfies both the S3 and S4 suites.
    """

    rule_name: str
    predicates: list
    result: str
    reason: str
    timestamp: str
    approval_trace: str
    latency_ms: float = 0.0
    budget_violated: bool = False

    def __getitem__(self, key: str) -> Any:
        """Return the named field, mapping-style, or raise ``KeyError``."""
        if key not in {f.name for f in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this entry."""
        return asdict(self)


class DataContractGate:
    """Validates a training-data record against a fixed schema contract (C13).

    Contract: every field in ``required_fields`` must be present and non-null;
    every field named in ``ranges`` must fall within its declared inclusive
    ``[lo, hi]`` interval.
    """

    RULE_NAME = "data_contract_v1"

    def __init__(
        self,
        required_fields: Sequence[str],
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.required_fields = list(required_fields)
        self.ranges = dict(ranges or {})

    def evaluate(self, record: dict) -> GateDecision:
        """Evaluate ``record`` against the contract, returning a decision."""
        predicates: List[Dict[str, Any]] = []
        for name in self.required_fields:
            present = name in record and record[name] is not None
            predicates.append({"predicate": f"has_field:{name}", "result": present})
            if not present:
                return GateDecision(
                    rule_name=self.RULE_NAME,
                    passed=False,
                    reason=f"Missing or null required field: {name}",
                    predicates=predicates,
                    timestamp=now_iso(),
                )
        for name, (lo, hi) in self.ranges.items():
            value = record.get(name)
            in_range = value is not None and lo <= value <= hi
            predicates.append(
                {"predicate": f"in_range:{name}[{lo},{hi}]", "result": in_range}
            )
            if not in_range:
                return GateDecision(
                    rule_name=self.RULE_NAME,
                    passed=False,
                    reason=f"Field {name}={value!r} outside contract range [{lo},{hi}]",
                    predicates=predicates,
                    timestamp=now_iso(),
                )
        return GateDecision(
            rule_name=self.RULE_NAME,
            passed=True,
            reason="All contract predicates satisfied",
            predicates=predicates,
            timestamp=now_iso(),
        )


class PolicyEngine:
    """Unified gate evaluator: structured audit logging (C15) + latency budget (C16).

    Wraps any gate object exposing ``.evaluate(record) -> GateDecision``. Each
    call records the decision in an append-only audit log and measures the
    wall-clock latency of the evaluation against ``budget_ms``.
    """

    def __init__(self, budget_ms: float = 100.0) -> None:
        self.budget_ms = budget_ms
        self.audit_log: List[AuditEntry] = []

    def evaluate(
        self, gate, record: dict, actor: str = "policy_engine"
    ) -> GateDecision:
        """Evaluate ``gate`` on ``record``, audit it and time it."""
        start = time.perf_counter()
        decision = gate.evaluate(record)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        decision.latency_ms = elapsed_ms
        decision.budget_violated = elapsed_ms > self.budget_ms
        entry = AuditEntry(
            rule_name=decision.rule_name,
            predicates=decision.predicates,
            result="pass" if decision.passed else "fail",
            reason=decision.reason,
            timestamp=decision.timestamp,
            approval_trace=f"evaluated_by:{actor}",
            latency_ms=elapsed_ms,
            budget_violated=decision.budget_violated,
        )
        self.audit_log.append(entry)
        return decision

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return the audit log as a JSON-serializable list of dicts."""
        return [e.to_dict() for e in self.audit_log]
