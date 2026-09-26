"""
Model-quality gate against a registered baseline (Claim C14).

Maps to requirement R2.1 (CI/CD Policy Gates) and is lifted from the frozen
reference implementation in ``tests/R2_1_S2.py``.

C14: "Model-quality gates reject models underperforming the baseline by
> threshold. Measured as gate accuracy (true-accept + true-reject) >= 95% on
held-out test set."

``ModelQualityGate`` shares the unified ``GateDecision``/``now_iso`` surface
defined in :mod:`mlops.gates`. ``EvaluationMetrics`` and ``ModelComparator``
are the supporting types named in plan-a; the gate's decision logic itself
matches R2_1_S2 exactly, including the inclusive threshold boundary.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from mlops.gates import GateDecision, now_iso

__all__ = [
    "BaselineMetrics",
    "EvaluationMetrics",
    "BaselineStore",
    "ModelQualityGate",
    "ModelComparator",
]


@dataclass
class BaselineMetrics:
    """The metrics a candidate model is compared against."""

    accuracy: float


@dataclass
class EvaluationMetrics:
    """A candidate model's evaluation metrics (accuracy plus supporting fields)."""

    accuracy: float
    precision: float = 0.0
    recall: float = 0.0


class BaselineStore:
    """In-memory registry of baseline metrics keyed by model family."""

    def __init__(self) -> None:
        self._baselines: Dict[str, BaselineMetrics] = {}

    def set_baseline(self, model_family: str, metrics: BaselineMetrics) -> None:
        """Register ``metrics`` as the baseline for ``model_family``."""
        self._baselines[model_family] = metrics

    def get_baseline(self, model_family: str) -> BaselineMetrics:
        """Return the baseline for ``model_family`` or raise ``KeyError``."""
        if model_family not in self._baselines:
            raise KeyError(f"No baseline registered for model family: {model_family}")
        return self._baselines[model_family]


class ModelQualityGate:
    """Rejects a candidate that degrades accuracy vs. its baseline by more than
    ``threshold`` (inclusive boundary: exactly ``threshold`` still passes).
    """

    RULE_NAME = "model_quality_v1"

    def __init__(self, baseline_store: BaselineStore, threshold: float = 0.02) -> None:
        self.store = baseline_store
        self.threshold = threshold

    def evaluate(self, model_family: str, candidate_accuracy: float) -> GateDecision:
        """Evaluate a candidate accuracy against its family baseline."""
        baseline = self.store.get_baseline(model_family)
        degradation = round(baseline.accuracy - candidate_accuracy, 9)
        passed = degradation <= self.threshold
        predicates = [
            {
                "predicate": f"accuracy_degradation<={self.threshold}",
                "result": passed,
                "observed_degradation": round(degradation, 6),
            }
        ]
        reason = (
            f"candidate accuracy {candidate_accuracy} within {self.threshold} "
            f"of baseline {baseline.accuracy}"
            if passed
            else f"candidate accuracy {candidate_accuracy} degrades "
            f"{degradation:.4f} > threshold {self.threshold}"
        )
        return GateDecision(
            rule_name=self.RULE_NAME,
            passed=passed,
            reason=reason,
            predicates=predicates,
            timestamp=now_iso(),
        )


class ModelComparator:
    """Compare a candidate's evaluation metrics against a baseline's.

    Returns the signed accuracy delta (candidate - baseline) and a boolean
    ``regressed`` flag, set when the candidate falls more than ``threshold``
    below the baseline.
    """

    def __init__(self, threshold: float = 0.02) -> None:
        self.threshold = threshold

    def compare(
        self,
        candidate: EvaluationMetrics,
        baseline: EvaluationMetrics,
    ) -> Dict[str, object]:
        """Return ``{"delta": candidate - baseline, "regressed": bool}``."""
        delta = candidate.accuracy - baseline.accuracy
        regressed = delta < -self.threshold
        return {"delta": delta, "regressed": regressed}
