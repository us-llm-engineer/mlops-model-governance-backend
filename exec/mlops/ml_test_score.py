"""
ML Test Score evaluator: 28-item rubric for ML production readiness (C79).

Implements the Breck et al. "The ML Test Score: A Rubric for ML Production
Readiness and Technical Debt Reduction" (2017). The rubric organizes 28 tests
into four categories (7 per category): data, model, infrastructure, monitoring.

Each item scores 0 (missing/False), 0.5 (manual, value 0.5), or 1 (automated,
True/1). Category score = sum of item scores. Final score = minimum category
score (Breck rubric: the weakest category determines production readiness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["ITEMS", "ITEMS_BY_CATEGORY", "ScoreReport", "evaluate_ml_test_score", "score_to_context"]


@dataclass
class ScoreReport:
    """Result of an ML Test Score evaluation."""

    total: float  # Sum of all item scores (0-28)
    by_category: Dict[str, float] = field(default_factory=dict)  # category -> sum
    final: float = 0.0  # Minimum category score
    missing: List[str] = field(default_factory=list)  # Item ids with missing evidence


# 28-item rubric from Breck et al. "The ML Test Score" (2017), 7 per category
ITEMS: List[Dict[str, Any]] = [
    # Data Tests (7 items)
    {
        "id": "feature_expectations_schema",
        "category": "data",
        "text": "Feature expectations captured in schema (presence, type, range, distribution)",
        "source": "breck-2017",
    },
    {
        "id": "features_beneficial",
        "category": "data",
        "text": "All input features are beneficial to model (no unused features)",
        "source": "breck-2017",
    },
    {
        "id": "feature_cost_vs_benefit",
        "category": "data",
        "text": "Feature cost does not exceed benefit",
        "source": "breck-2017",
    },
    {
        "id": "feature_meta_requirements",
        "category": "data",
        "text": "Meta-requirements (labeling, metadata, SLA) are documented",
        "source": "breck-2017",
    },
    {
        "id": "data_pipeline_privacy",
        "category": "data",
        "text": "Data pipeline has appropriate privacy controls",
        "source": "breck-2017",
    },
    {
        "id": "new_features_added_quickly",
        "category": "data",
        "text": "New features can be added and deployed quickly",
        "source": "breck-2017",
    },
    {
        "id": "input_feature_code_tested",
        "category": "data",
        "text": "Training input feature code is version controlled and tested",
        "source": "breck-2017",
    },
    # Model Tests (7 items)
    {
        "id": "model_spec_reviewed",
        "category": "model",
        "text": "Model specification is reviewed and version controlled",
        "source": "breck-2017",
    },
    {
        "id": "offline_online_metrics_correlate",
        "category": "model",
        "text": "Offline and online metrics are correlated",
        "source": "breck-2017",
    },
    {
        "id": "hyperparameters_tuned",
        "category": "model",
        "text": "All hyperparameters are tuned",
        "source": "breck-2017",
    },
    {
        "id": "staleness_impact_known",
        "category": "model",
        "text": "Impact of model staleness on performance is known",
        "source": "breck-2017",
    },
    {
        "id": "simpler_model_not_better",
        "category": "model",
        "text": "Simpler baseline model is outperformed",
        "source": "breck-2017",
    },
    {
        "id": "quality_on_data_slices",
        "category": "model",
        "text": "Quality metrics are evaluated on important data slices",
        "source": "breck-2017",
    },
    {
        "id": "inclusion_tested",
        "category": "model",
        "text": "Model inclusion requirements tested (no fairness regressions)",
        "source": "breck-2017",
    },
    # Infrastructure Tests (7 items)
    {
        "id": "training_reproducible",
        "category": "infrastructure",
        "text": "Training pipeline is fully reproducible",
        "source": "breck-2017",
    },
    {
        "id": "model_spec_unit_tested",
        "category": "infrastructure",
        "text": "Model serving code is unit tested",
        "source": "breck-2017",
    },
    {
        "id": "pipeline_integration_tested",
        "category": "infrastructure",
        "text": "Training and serving code paths are tested for consistency",
        "source": "breck-2017",
    },
    {
        "id": "quality_validated_before_serving",
        "category": "infrastructure",
        "text": "Quality metrics are validated before model serving",
        "source": "breck-2017",
    },
    {
        "id": "model_debuggable",
        "category": "infrastructure",
        "text": "Model is debuggable (can inspect predictions and intermediate values)",
        "source": "breck-2017",
    },
    {
        "id": "models_canaried_before_serving",
        "category": "infrastructure",
        "text": "Models are canaried before serving to all users",
        "source": "breck-2017",
    },
    {
        "id": "serving_rollback_possible",
        "category": "infrastructure",
        "text": "Model rollback is possible on all platforms",
        "source": "breck-2017",
    },
    # Monitoring Tests (7 items)
    {
        "id": "dependency_change_notification",
        "category": "monitoring",
        "text": "Dependency change notifications trigger alerts",
        "source": "breck-2017",
    },
    {
        "id": "data_invariants_hold",
        "category": "monitoring",
        "text": "Data invariants hold in training and serving",
        "source": "breck-2017",
    },
    {
        "id": "training_serving_not_skewed",
        "category": "monitoring",
        "text": "Training and serving features are not skewed",
        "source": "breck-2017",
    },
    {
        "id": "models_not_too_stale",
        "category": "monitoring",
        "text": "Models are not too stale (staleness triggers retraining)",
        "source": "breck-2017",
    },
    {
        "id": "numerically_stable",
        "category": "monitoring",
        "text": "Predictions are numerically stable (no catastrophic cancellation)",
        "source": "breck-2017",
    },
    {
        "id": "compute_performance_not_regressed",
        "category": "monitoring",
        "text": "Compute performance does not regress",
        "source": "breck-2017",
    },
    {
        "id": "prediction_quality_not_regressed",
        "category": "monitoring",
        "text": "Prediction quality does not regress",
        "source": "breck-2017",
    },
]


# Helper: group items by category
ITEMS_BY_CATEGORY: Dict[str, List[Dict[str, Any]]] = {}
for item in ITEMS:
    category = item["category"]
    if category not in ITEMS_BY_CATEGORY:
        ITEMS_BY_CATEGORY[category] = []
    ITEMS_BY_CATEGORY[category].append(item)


def evaluate_ml_test_score(evidence: Dict[str, Any]) -> ScoreReport:
    """Evaluate evidence against the ML Test Score rubric.

    Args:
        evidence: dict of item_id -> score (0, 0.5, 1) or bool/None.
                 Missing keys, None, or False score 0;
                 True or 1 scores 1; 0.5 scores 0.5.

    Returns:
        ScoreReport with total, by_category, final (minimum category), missing.

    The final score follows Breck's rubric: minimum category score determines
    production readiness (a weak category in any dimension is the bottleneck).
    """
    by_category = {}
    missing = []

    for item in ITEMS:
        item_id = item["id"]
        category = item["category"]
        value = evidence.get(item_id)

        # Score logic: missing/None/False -> 0, True/1 -> 1, 0.5 -> 0.5
        if value is None or value is False:
            score = 0.0
        elif value is True or value == 1:
            score = 1.0
        elif value == 0.5:
            score = 0.5
        elif isinstance(value, (int, float)) and value in (0, 0.5, 1):
            score = float(value)
        else:
            # Invalid value: treat as missing
            score = 0.0

        if score == 0.0 and value is None:
            missing.append(item_id)

        by_category.setdefault(category, 0.0)
        by_category[category] += score

    # Final score = minimum category score (Breck rubric)
    final = min(by_category.values()) if by_category else 0.0

    return ScoreReport(
        total=sum(by_category.values()),
        by_category=by_category,
        final=final,
        missing=missing,
    )


def score_to_context(report: ScoreReport) -> Dict[str, Any]:
    """Convert a ScoreReport to a policy context dict.

    Exposes ml_test_score and per-category scores for policy rules.
    """
    return {
        "ml_test_score": report.final,
        "ml_test_score_data": report.by_category.get("data", 0.0),
        "ml_test_score_model": report.by_category.get("model", 0.0),
        "ml_test_score_infrastructure": report.by_category.get("infrastructure", 0.0),
        "ml_test_score_monitoring": report.by_category.get("monitoring", 0.0),
        "ml_test_score_total": report.total,
    }
