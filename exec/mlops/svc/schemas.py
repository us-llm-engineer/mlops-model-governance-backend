"""F1.4 pydantic v2 schema layer for MLOps control plane (mlops.svc.schemas).

All models use strict mode: no type coercion, no extra fields, frozen.
Strings: 1-200 chars, no control characters. Floats must be finite.
"""

import dataclasses
import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    ValidationError,
)

from mlops.kernel import ValidationFailed
from mlops.policy_engine import PolicyBundle, Rule

__all__ = [
    "RuleModel",
    "PolicyBundleModel",
    "LintFindingModel",
    "DriftReportModel",
    "IncidentModel",
    "export_json_schemas",
    "parse_model",
]


def _validate_string_field(value: str, min_len: int = 1, max_len: int = 200) -> str:
    """Validate a string field: non-empty, max length, no control characters."""
    if not isinstance(value, str):
        raise ValueError("must be a string")
    if len(value) < min_len:
        raise ValueError(f"must be at least {min_len} character(s)")
    if len(value) > max_len:
        raise ValueError(f"must be at most {max_len} characters")
    # Check for control characters (ASCII 0-31 and 127)
    for ch in value:
        if ord(ch) < 32 or ord(ch) == 127:
            raise ValueError("contains control characters")
    return value


def _validate_finite_float(value: float) -> float:
    """Validate that a float is finite (not NaN, inf, or -inf)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("must be numeric")
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError("must be finite (not NaN or inf)")
    return fv


def _validate_params_depth(obj: Any, max_depth: int = 8, current_depth: int = 0) -> bool:
    """Check if object nesting depth is within bounds.

    Depth counts only container (dict/list) nesting levels: a scalar leaf
    value does not consume an extra level of depth. This matches the
    contract's "depth<=8" (a dict nested exactly 8 levels deep must be
    accepted, not rejected).
    """
    if current_depth > max_depth:
        return False
    if isinstance(obj, dict):
        # Check keys are strings
        for k, v in obj.items():
            if not isinstance(k, str):
                return False
            # Check values for finite floats (no NaN/inf)
            if isinstance(v, float):
                if not math.isfinite(v):
                    return False
            if isinstance(v, (dict, list)):
                if not _validate_params_depth(v, max_depth, current_depth + 1):
                    return False
    elif isinstance(obj, list):
        for item in obj:
            # Check items for finite floats
            if isinstance(item, float):
                if not math.isfinite(item):
                    return False
            if isinstance(item, (dict, list)):
                if not _validate_params_depth(item, max_depth, current_depth + 1):
                    return False
    return True


class RuleModel(BaseModel):
    """A policy rule (frozen, strict, no extra fields)."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    id: str
    version: int
    kind: str
    params: Optional[Dict[str, Any]] = None
    severity: Literal["block", "warn"] = "block"

    @field_validator("id", "kind", mode="before")
    @classmethod
    def validate_string_fields(cls, v):
        """Validate string fields: 1-200 chars, no control characters."""
        return _validate_string_field(v, min_len=1, max_len=200)

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v):
        """Validate version: int >= 1."""
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("must be an integer")
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("params", mode="before")
    @classmethod
    def validate_params(cls, v):
        """Validate params: JSON-like dict with string keys, depth <= 8, no NaN/inf."""
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("must be a dict or null")
        if not _validate_params_depth(v):
            raise ValueError("must be JSON-like with depth <= 8, string keys, and finite floats")
        return v


class PolicyBundleModel(BaseModel):
    """A versioned collection of policy rules."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: str
    version: int
    rules: List[RuleModel]

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v):
        """Validate name: 1-200 chars, no control characters."""
        return _validate_string_field(v, min_len=1, max_len=200)

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v):
        """Validate version: int >= 1."""
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("must be an integer")
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("rules", mode="before")
    @classmethod
    def validate_rules(cls, v):
        """Validate rules: list with 1-200 items."""
        if not isinstance(v, list):
            raise ValueError("must be a list")
        if len(v) < 1:
            raise ValueError("must have at least 1 rule")
        if len(v) > 200:
            raise ValueError("must have at most 200 rules")
        return v

    def to_domain(self) -> PolicyBundle:
        """Convert to a domain PolicyBundle."""
        rules = [
            Rule(
                id=r.id,
                version=r.version,
                kind=r.kind,
                params=r.params,
                severity=r.severity,
            )
            for r in self.rules
        ]
        return PolicyBundle(
            name=self.name,
            version=self.version,
            rules=rules,
        )

    @classmethod
    def from_domain(cls, bundle: PolicyBundle) -> "PolicyBundleModel":
        """Reconstruct from a domain PolicyBundle."""
        rules = [
            RuleModel(
                id=r.id,
                version=r.version,
                kind=r.kind,
                params=r.params,
                severity=r.severity,
            )
            for r in bundle.rules
        ]
        return cls(
            name=bundle.name,
            version=bundle.version,
            rules=rules,
        )


class LintFindingModel(BaseModel):
    """A single linting finding on a Kubernetes manifest."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    rule_id: str
    category: str
    subcategory: str
    severity: str
    path: str
    message: str
    weight: float

    @classmethod
    def from_domain(cls, finding: Any) -> "LintFindingModel":
        """Construct from a domain Finding dataclass."""
        if not dataclasses.is_dataclass(finding):
            raise ValidationFailed("Finding must be a dataclass instance")
        return cls(
            rule_id=finding.rule_id,
            category=finding.category,
            subcategory=finding.subcategory,
            severity=finding.severity,
            path=finding.path,
            message=finding.message,
            weight=finding.weight,
        )


class DriftReportModel(BaseModel):
    """A drift monitoring report."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    level: str
    psi: float
    ks: float
    n: int
    window_full: bool
    ks_pvalue: float = 1.0
    psi_adjusted: float = 0.0

    @classmethod
    def from_domain(cls, report: Any) -> "DriftReportModel":
        """Construct from a domain DriftReport dataclass."""
        if not dataclasses.is_dataclass(report):
            raise ValidationFailed("DriftReport must be a dataclass instance")
        return cls(
            level=report.level,
            psi=report.psi,
            ks=report.ks,
            n=report.n,
            window_full=report.window_full,
            ks_pvalue=report.ks_pvalue,
            psi_adjusted=report.psi_adjusted,
        )


class IncidentModel(BaseModel):
    """An incident record (selected fields)."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    incident_id: str
    title: str
    severity: str
    status: str
    opened_ts: float
    deadline: float
    current_tier: int
    steps_completed: List[str]

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "IncidentModel":
        """Construct from an incident record dict (ignores extra keys)."""
        # Extract only the fields we need; ignore extra keys like actor, signal, postmortem
        required_keys = {
            "incident_id",
            "title",
            "severity",
            "status",
            "opened_ts",
            "deadline",
            "current_tier",
            "steps_completed",
        }

        # Check that all required keys are present
        missing_keys = required_keys - set(record.keys())
        if missing_keys:
            raise ValidationFailed(f"Missing required fields: {', '.join(sorted(missing_keys))}")

        # Extract only the fields we need
        data = {k: record[k] for k in required_keys}

        # The severity field in the record is an Enum; extract its name
        severity_val = record["severity"]
        if hasattr(severity_val, "name"):
            data["severity"] = severity_val.name

        return cls(**data)


def export_json_schemas() -> Dict[str, Dict[str, Any]]:
    """Export JSON schemas for all models in deterministic order."""
    schemas = {
        "PolicyBundle": PolicyBundleModel.model_json_schema(),
        "Rule": RuleModel.model_json_schema(),
        "LintFinding": LintFindingModel.model_json_schema(),
        "DriftReport": DriftReportModel.model_json_schema(),
        "Incident": IncidentModel.model_json_schema(),
    }
    return schemas


def parse_model(model_cls: type, raw: Dict[str, Any]) -> BaseModel:
    """Parse raw dict into a model, converting pydantic ValidationError to ValidationFailed.

    The error message names the offending field but never echoes the value.
    """
    try:
        return model_cls(**raw)
    except ValidationError as e:
        # Extract field names from the validation error
        # e.errors() returns a list of dicts with 'loc', 'msg', 'type', etc.
        field_names = []
        for error in e.errors():
            loc = error.get("loc")
            if loc and len(loc) > 0:
                field_names.append(str(loc[0]))

        if field_names:
            field_str = ", ".join(sorted(set(field_names)))
            raise ValidationFailed(f"Invalid field(s): {field_str}")
        else:
            raise ValidationFailed("Validation failed")
