"""
MLOps interop framework: bridges to external systems for cross-check evaluation.

This module provides adapters that translate MLOps domain objects to external
systems (e.g., casbin for RBAC) to enable differential testing and verification
that our implementations match the external system's behavior.
"""

from mlops.interop.casbin_adapter import (
    CASBIN_MODEL_TEXT,
    build_enforcer_from_bundle,
    enforcer_decide,
    DifferentialResult,
    run_differential,
    generate_requests,
)
from mlops.interop.mlflow_bridge import MlflowBridge

__all__ = [
    "CASBIN_MODEL_TEXT",
    "build_enforcer_from_bundle",
    "enforcer_decide",
    "DifferentialResult",
    "run_differential",
    "generate_requests",
    "MlflowBridge",
]
