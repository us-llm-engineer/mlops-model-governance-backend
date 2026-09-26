"""F1.4 slim suite: pydantic v2 schema layer (mlops.svc.schemas), contract in
the API contract section 4.

Frozen expectations (contract-derived):
  All models: ConfigDict(strict=True, extra="forbid", frozen=True).
  Strings: 1..200 chars, no control characters. Floats finite.
  RuleModel(id:str, version:int>=1, kind:str, params:dict|None, severity:Literal["block","warn"]="block");
    params: JSON-like, depth<=8, keys str, no NaN/inf.
  PolicyBundleModel(name:str, version:int>=1, rules:list[RuleModel] (1..200))
    .to_domain() -> mlops.policy_engine.PolicyBundle; classmethod .from_domain(bundle);
    round trip preserves bundle.hash.
  LintFindingModel.from_domain(finding); DriftReportModel.from_domain(report) copy fields exactly.
  IncidentModel.from_record(record: dict) projects the known subset of a real incident record
    (records carry extra keys like actor/signal/postmortem that must be ignored, not forbidden).
  export_json_schemas() -> dict[str, dict] with exactly keys
    "PolicyBundle","Rule","LintFinding","DriftReport","Incident" in that order -> Model.model_json_schema();
    deterministic key order.
  parse_model(model_cls, raw) -> model; on invalid input raises mlops.kernel.ValidationFailed whose
    message names the offending field but never echoes the offending value.
  `import mlops` must not import mlops.svc.schemas (pydantic-only svc modules stay out of the core import).
"""
import dataclasses
import json
import os
import subprocess
import sys

import pydantic
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

from mlops.audit import ChainedAuditStore
from mlops.config_lint import Finding
from mlops.drift import DriftReport
from mlops.incidents import IncidentManager
from mlops.kernel import ManualClock, ValidationFailed
from mlops.policy_engine import PolicyBundle, PolicyDecisionPoint, Rule
from mlops.svc.schemas import (
    DriftReportModel,
    IncidentModel,
    LintFindingModel,
    PolicyBundleModel,
    RuleModel,
    export_json_schemas,
    parse_model,
)

NAN = float("nan")
INF = float("inf")
SECRET_VALUE = "s3cr3t-token-xyz"


def _rule_kwargs(**over):
    base = dict(id="r1", version=1, kind="min_metric", params={"metric": "acc", "min": 0.9}, severity="block")
    base.update(over)
    return base


def _nested(depth):
    """Build a dict nested `depth` levels deep under key 'nested'."""
    d = {"v": 0}
    for _ in range(depth):
        d = {"nested": d}
    return d


def _incident_record():
    clock = ManualClock(start=1000.0)
    audit = ChainedAuditStore(b"incident-secret-key-00")
    mgr = IncidentManager(audit, clock)
    iid = mgr.open("alice", "prod accuracy drop", {"accuracy_drop_pct": 15.0})
    return mgr.get(iid)


# 1 -- strict-mode type rejections (int-as-str, bool-for-int, NaN, inf, extra field) -----
@pytest.mark.parametrize("kwargs", [
    _rule_kwargs(version="1"),
    _rule_kwargs(version=True),
    _rule_kwargs(params={"metric": "acc", "min": NAN}),
    _rule_kwargs(params={"metric": "acc", "min": INF}),
    _rule_kwargs(unexpected_field="oops"),
])
def test_rule_model_strict_mode_rejects_bad_types(kwargs):
    with pytest.raises(pydantic.ValidationError):
        RuleModel(**kwargs)


# 2 -- string limits: empty, >200 chars, control characters -----------------------------
@pytest.mark.parametrize("kwargs", [
    _rule_kwargs(id=""),
    _rule_kwargs(id="x" * 201),
    _rule_kwargs(id="ab\x00cd"),
])
def test_rule_model_string_limits(kwargs):
    with pytest.raises(pydantic.ValidationError):
        RuleModel(**kwargs)


# 3 -- params depth > 8 and non-str keys rejected ----------------------------------------
@pytest.mark.parametrize("params", [
    _nested(12),
    {1: "x"},
])
def test_rule_model_params_constraints(params):
    with pytest.raises(pydantic.ValidationError):
        RuleModel(**_rule_kwargs(params=params))


# 4 -- rules list length bounds: 0 and 201 items -----------------------------------------
@pytest.mark.parametrize("n_rules", [0, 201])
def test_policy_bundle_model_rules_length_bounds(n_rules):
    rules = [RuleModel(**_rule_kwargs(id=f"r{i}")) for i in range(n_rules)]
    with pytest.raises(pydantic.ValidationError):
        PolicyBundleModel(name="p", version=1, rules=rules)


# 5 -- severity Literal rejects invalid values -------------------------------------------
@pytest.mark.parametrize("kwargs", [
    _rule_kwargs(severity="critical"),
    _rule_kwargs(severity=1),
])
def test_rule_model_severity_literal_rejects_invalid(kwargs):
    with pytest.raises(pydantic.ValidationError):
        RuleModel(**kwargs)


# 6 -- frozen: assignment after construction raises --------------------------------------
def test_rule_model_frozen_assignment_raises():
    model = RuleModel(**_rule_kwargs())
    with pytest.raises(pydantic.ValidationError):
        model.id = "changed"


# 7 -- PolicyBundleModel round trip preserves domain hash --------------------------------
def test_policy_bundle_round_trip_preserves_hash():
    rules = [
        Rule(id="r1", version=1, kind="min_metric", params={"metric": "acc", "min": 0.9}, severity="block"),
        Rule(id="r2", version=2, kind="flag_true", params={"key": "ok"}, severity="warn"),
    ]
    bundle = PolicyBundle(rules=rules, name="p", version=3)
    model = PolicyBundleModel.from_domain(bundle)
    back = model.to_domain()
    assert back.hash == bundle.hash


# 8 -- to_domain() builds a real, usable PolicyBundle ------------------------------------
def test_policy_bundle_to_domain_usable_by_decision_point():
    rule = Rule(id="r1", version=1, kind="min_metric", params={"metric": "acc", "min": 0.9}, severity="block")
    bundle = PolicyBundle(rules=[rule], name="p", version=1)
    model = PolicyBundleModel.from_domain(bundle)
    dp = PolicyDecisionPoint(model.to_domain())
    assert dp.decide("promote", {"acc": 0.95}).allow is True
    assert dp.decide("promote", {"acc": 0.5}).allow is False


# 9 -- LintFindingModel.from_domain copies fields exactly --------------------------------
def test_lint_finding_from_domain_copies_fields_exactly():
    finding = Finding(
        rule_id="R1", category="security", subcategory="privileged", severity="block",
        path="spec.template.spec.containers[0]", message="privileged container", weight=1.5,
    )
    model = LintFindingModel.from_domain(finding)
    for f in dataclasses.fields(Finding):
        assert getattr(model, f.name) == getattr(finding, f.name)


# 10 -- LintFindingModel.from_domain rejects non-dataclass input -------------------------
def test_lint_finding_from_domain_rejects_non_dataclass():
    with pytest.raises(ValidationFailed):
        LintFindingModel.from_domain({"rule_id": "R1"})


# 11 -- DriftReportModel.from_domain copies fields exactly -------------------------------
def test_drift_report_from_domain_copies_fields_exactly():
    report = DriftReport(level="warn", psi=0.15, ks=0.05, n=250, window_full=True, ks_pvalue=0.02, psi_adjusted=0.12)
    model = DriftReportModel.from_domain(report)
    for f in dataclasses.fields(DriftReport):
        assert getattr(model, f.name) == getattr(report, f.name)


# 12 -- DriftReportModel.from_domain rejects non-dataclass input -------------------------
def test_drift_report_from_domain_rejects_non_dataclass():
    with pytest.raises(ValidationFailed):
        DriftReportModel.from_domain({"level": "ok"})


# 13 -- IncidentModel.from_record on a real record ----------------------------------------
def test_incident_model_from_record_valid():
    record = _incident_record()
    model = IncidentModel.from_record(record)
    assert model.incident_id == record["incident_id"]
    assert model.status == "open"
    assert model.severity == "P1"
    assert model.steps_completed == []
    assert all(isinstance(s, str) for s in model.steps_completed)


# 14 -- IncidentModel.from_record on a record missing required keys ----------------------
@pytest.mark.parametrize("missing_key", ["incident_id", "severity", "steps_completed"])
def test_incident_model_from_record_missing_keys(missing_key):
    record = _incident_record()
    del record[missing_key]
    with pytest.raises(ValidationFailed):
        IncidentModel.from_record(record)


# 15 -- export_json_schemas() exact keys, order, and shape --------------------------------
def test_export_json_schemas_keys_and_properties():
    schemas = export_json_schemas()
    assert list(schemas.keys()) == ["PolicyBundle", "Rule", "LintFinding", "DriftReport", "Incident"]
    for name, schema in schemas.items():
        assert isinstance(schema, dict)
        assert "properties" in schema


# 16 -- export_json_schemas() JSON dump is deterministic across calls --------------------
def test_export_json_schemas_deterministic_dumps():
    first = json.dumps(export_json_schemas())
    second = json.dumps(export_json_schemas())
    assert first == second


# 17 -- parse_model returns the model on valid input --------------------------------------
def test_parse_model_valid_returns_model():
    model = parse_model(RuleModel, _rule_kwargs())
    assert isinstance(model, RuleModel)
    assert model.id == "r1"


# 18 -- parse_model error names the field but never echoes the value ---------------------
def test_parse_model_error_hides_value_but_names_field():
    raw = _rule_kwargs(version=SECRET_VALUE)
    with pytest.raises(ValidationFailed) as exc_info:
        parse_model(RuleModel, raw)
    message = str(exc_info.value)
    assert SECRET_VALUE not in message
    assert "version" in message


# 19 -- `import mlops` never pulls in mlops.svc.schemas (pydantic-only svc module) --------
def test_import_mlops_does_not_import_svc_schemas():
    code = (
        "import sys\n"
        "import mlops\n"
        "assert 'mlops.svc.schemas' not in sys.modules\n"
    )
    exec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec")
    env = dict(os.environ)
    env["PYTHONPATH"] = exec_dir
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=exec_dir, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
