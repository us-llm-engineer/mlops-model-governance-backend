"""F1.3 slim suite: exec/mlops/svc/metrics_collector.py, contract in
the API contract section 3 (metrics_collector).

Frozen expectations (contract-derived):
  DomainMetricsCollector(*, audit=None, policy_store=None, incidents=None, models=None):
  each provider optional; MUST implement describe() (TRAP: a Collector without describe()
  is not duplicate-checked by prometheus and would emit every series twice on double
  registration). collect() yields, only for providers actually given:
    mlops_audit_chain_length          gauge = audit.head()["length"]
    mlops_policy_active_version       gauge = policy_store.active()[0], 0 when none active
                                       (NotFound/Conflict caught)
    mlops_policy_versions             gauge = len(policy_store.history())
    mlops_incidents_open{severity=}   gauge from incidents.open_counts() -> dict[str,int];
                                       P0/P1/P2/P3 always present
    mlops_collector_errors{collector=} counter; a provider that raises must NOT fail the
                                       scrape -- its family is omitted, the counter for its
                                       name is bumped, and the exception text is never
                                       exported.
  register_domain_metrics(registry, **providers) -> DomainMetricsCollector; registry.register;
  a second call on the same registry raises ValueError (prometheus DuplicateTimeseries is a
  ValueError subclass). registry must be a prometheus_client.CollectorRegistry, else
  mlops.kernel.ValidationFailed. Collector names appearing in the "collector" label are only
  ever audit/policy/incidents/models; severity label values are only ever P0/P1/P2/P3.
  `import mlops` (subprocess) never imports prometheus_client.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec"))

import subprocess

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from mlops import kernel
from mlops.audit import ChainedAuditStore
from mlops.policy_engine import PolicyBundle, Rule
from mlops.policy_store import PolicyStore
from mlops.svc.metrics_collector import DomainMetricsCollector, register_domain_metrics

SECRET = b"f1-3-metrics-secret"
ALLOWED_LABEL_VALUES = {"P0", "P1", "P2", "P3", "audit", "policy", "incidents", "models"}
RAISE_SECRET = "TOTALLY-SECRET-EXCEPTION-TEXT-9f8e7d"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def R(rid, kind="min_metric", params=None, sev="block"):
    return Rule(id=rid, version=1, kind=kind, params=params or {"metric": "acc", "min": 0.5}, severity=sev)


def B(*rules, name="p", version=0):
    return PolicyBundle(rules=list(rules) or [R("r1")], name=name, version=version)


def make_policy_store(n_publishes=0, activate=False):
    """A real PolicyStore with n_publishes versions, optionally with the last activated."""
    ps = PolicyStore(audit=ChainedAuditStore(SECRET))
    version = None
    for i in range(n_publishes):
        version = ps.publish("admin", B(R(f"r{i}", params={"metric": "acc", "min": 0.5 + i * 0.01})))
    if activate and version is not None:
        ps.activate("admin", version)
    return ps


def make_audit(n_appends=0):
    audit = ChainedAuditStore(SECRET)
    for i in range(n_appends):
        audit.append("actor", "action", "res", "allow")
    return audit


class FakeIncidents:
    """Simple duck-typed incidents provider: open_counts() -> dict[str, int]."""

    def __init__(self, counts):
        self._counts = dict(counts)

    def open_counts(self):
        return dict(self._counts)


class RaisingAudit:
    def head(self):
        raise RuntimeError("audit boom " + RAISE_SECRET)


class RaisingPolicyStore:
    def active(self):
        raise RuntimeError("policy boom " + RAISE_SECRET)

    def history(self):
        raise RuntimeError("policy boom " + RAISE_SECRET)


class RaisingIncidents:
    def open_counts(self):
        raise RuntimeError("incidents boom " + RAISE_SECRET)


def scrape(registry):
    return generate_latest(registry).decode()


def families(text):
    return list(text_string_to_metric_families(text))


def find_sample(text, name, labels=None):
    labels = labels or {}
    for fam in families(text):
        if fam.name != name:
            continue
        for s in fam.samples:
            if all(s.labels.get(k) == v for k, v in labels.items()):
                return s
    return None


def metric_value(text, name, labels=None):
    s = find_sample(text, name, labels)
    assert s is not None, f"no sample for {name} labels={labels} in:\n{text}"
    return s.value


def family_absent(text, name):
    return all(fam.name != name for fam in families(text))


def error_count(text, collector_name):
    s = find_sample(text, "mlops_collector_errors", {"collector": collector_name})
    return s.value if s is not None else 0.0


# ---------------------------------------------------------------------------
# audit family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 3, 7])
def test_audit_family_value_after_appends(n):
    registry = CollectorRegistry()
    register_domain_metrics(registry, audit=make_audit(n))
    assert metric_value(scrape(registry), "mlops_audit_chain_length") == float(n)


def test_audit_family_absent_when_no_audit_provider():
    registry = CollectorRegistry()
    register_domain_metrics(registry, policy_store=make_policy_store())
    assert family_absent(scrape(registry), "mlops_audit_chain_length")


# ---------------------------------------------------------------------------
# policy family
# ---------------------------------------------------------------------------

def test_policy_active_version_zero_when_none_active():
    registry = CollectorRegistry()
    register_domain_metrics(registry, policy_store=make_policy_store(n_publishes=2, activate=False))
    assert metric_value(scrape(registry), "mlops_policy_active_version") == 0.0


def test_policy_active_version_after_activation():
    registry = CollectorRegistry()
    register_domain_metrics(registry, policy_store=make_policy_store(n_publishes=1, activate=True))
    assert metric_value(scrape(registry), "mlops_policy_active_version") == 1.0


def test_policy_versions_count_matches_history_length():
    registry = CollectorRegistry()
    register_domain_metrics(registry, policy_store=make_policy_store(n_publishes=3, activate=False))
    text = scrape(registry)
    assert metric_value(text, "mlops_policy_versions") == 3.0
    assert metric_value(text, "mlops_policy_active_version") == 0.0


def test_policy_family_absent_when_no_policy_provider():
    registry = CollectorRegistry()
    register_domain_metrics(registry, audit=make_audit())
    text = scrape(registry)
    assert family_absent(text, "mlops_policy_active_version")
    assert family_absent(text, "mlops_policy_versions")


# ---------------------------------------------------------------------------
# incidents family
# ---------------------------------------------------------------------------

def test_incidents_open_counts_all_three_severities_present_and_correct():
    registry = CollectorRegistry()
    register_domain_metrics(registry, incidents=FakeIncidents({"P1": 2, "P2": 0, "P3": 5}))
    text = scrape(registry)
    assert metric_value(text, "mlops_incidents_open", {"severity": "P1"}) == 2.0
    assert metric_value(text, "mlops_incidents_open", {"severity": "P2"}) == 0.0
    assert metric_value(text, "mlops_incidents_open", {"severity": "P3"}) == 5.0


def test_incidents_family_absent_when_no_incidents_provider():
    registry = CollectorRegistry()
    register_domain_metrics(registry, audit=make_audit())
    assert family_absent(scrape(registry), "mlops_incidents_open")


# ---------------------------------------------------------------------------
# provider errors: isolated, non-fatal, never leaked
# ---------------------------------------------------------------------------

RAISING_CASES = [
    ("audit", RaisingAudit(), "mlops_audit_chain_length", "audit"),
    ("policy_store", RaisingPolicyStore(), "mlops_policy_active_version", "policy"),
    ("incidents", RaisingIncidents(), "mlops_incidents_open", "incidents"),
]


@pytest.mark.parametrize("kwarg,stub,absent_metric,label", RAISING_CASES)
def test_provider_raising_does_not_fail_scrape_omits_family_and_records_error(kwarg, stub, absent_metric, label):
    registry = CollectorRegistry()
    register_domain_metrics(registry, **{kwarg: stub})
    text = scrape(registry)
    assert family_absent(text, absent_metric)
    assert error_count(text, label) == 1.0
    assert RAISE_SECRET not in text


def test_error_counter_is_cumulative_across_scrapes():
    registry = CollectorRegistry()
    register_domain_metrics(registry, audit=RaisingAudit())
    first = scrape(registry)
    second = scrape(registry)
    assert error_count(first, "audit") == 1.0
    assert error_count(second, "audit") == 2.0


# ---------------------------------------------------------------------------
# register_domain_metrics / registry semantics
# ---------------------------------------------------------------------------

def test_register_domain_metrics_twice_same_registry_raises_valueerror():
    registry = CollectorRegistry()
    register_domain_metrics(
        registry,
        audit=make_audit(1),
        policy_store=make_policy_store(1, activate=True),
        incidents=FakeIncidents({"P1": 0, "P2": 0, "P3": 0}),
    )
    with pytest.raises(ValueError):
        register_domain_metrics(
            registry,
            audit=make_audit(1),
            policy_store=make_policy_store(1, activate=True),
            incidents=FakeIncidents({"P1": 0, "P2": 0, "P3": 0}),
        )


@pytest.mark.parametrize("bad_registry", [object(), "not-a-registry"])
def test_non_registry_argument_raises_validation_failed(bad_registry):
    with pytest.raises(kernel.ValidationFailed):
        register_domain_metrics(bad_registry, audit=make_audit())


def test_two_registries_stay_independent():
    registry_a = CollectorRegistry()
    registry_b = CollectorRegistry()
    register_domain_metrics(registry_a, audit=make_audit(2))
    register_domain_metrics(registry_b, audit=make_audit(9))
    assert metric_value(scrape(registry_a), "mlops_audit_chain_length") == 2.0
    assert metric_value(scrape(registry_b), "mlops_audit_chain_length") == 9.0


def test_returns_domain_metrics_collector_instance():
    registry = CollectorRegistry()
    collector = register_domain_metrics(registry, audit=make_audit())
    assert isinstance(collector, DomainMetricsCollector)


# ---------------------------------------------------------------------------
# exposition format and idempotence
# ---------------------------------------------------------------------------

def test_scrape_output_is_valid_exposition_text():
    registry = CollectorRegistry()
    register_domain_metrics(
        registry,
        audit=make_audit(4),
        policy_store=make_policy_store(2, activate=True),
        incidents=FakeIncidents({"P1": 1, "P2": 2, "P3": 3}),
    )
    text = scrape(registry)
    parsed = families(text)
    names = {fam.name for fam in parsed}
    assert "mlops_audit_chain_length" in names
    assert "mlops_incidents_open" in names


def test_collect_is_idempotent_across_two_scrapes_without_state_change():
    registry = CollectorRegistry()
    register_domain_metrics(
        registry,
        audit=make_audit(5),
        policy_store=make_policy_store(2, activate=True),
        incidents=FakeIncidents({"P1": 1, "P2": 0, "P3": 0}),
    )
    first = scrape(registry)
    second = scrape(registry)
    assert metric_value(first, "mlops_audit_chain_length") == metric_value(second, "mlops_audit_chain_length")
    assert metric_value(first, "mlops_policy_active_version") == metric_value(second, "mlops_policy_active_version")
    assert metric_value(first, "mlops_incidents_open", {"severity": "P1"}) == \
        metric_value(second, "mlops_incidents_open", {"severity": "P1"})


def test_all_families_absent_when_no_providers_given():
    registry = CollectorRegistry()
    register_domain_metrics(registry)
    text = scrape(registry)
    assert family_absent(text, "mlops_audit_chain_length")
    assert family_absent(text, "mlops_policy_active_version")
    assert family_absent(text, "mlops_policy_versions")
    assert family_absent(text, "mlops_incidents_open")


def test_models_provider_accepted_without_crashing():
    registry = CollectorRegistry()
    register_domain_metrics(registry, audit=make_audit(1), models=object())
    # Undocumented family for `models` in this contract slice; only requirement is no crash.
    scrape(registry)


# ---------------------------------------------------------------------------
# no user data ever appears in metric/label surface
# ---------------------------------------------------------------------------

def test_label_values_restricted_to_allowed_set():
    registry = CollectorRegistry()
    register_domain_metrics(
        registry,
        audit=make_audit(1),
        policy_store=make_policy_store(1, activate=True),
        incidents=FakeIncidents({"P1": 1, "P2": 2, "P3": 3}),
    )
    text = scrape(registry)
    for fam in families(text):
        for sample in fam.samples:
            for key, value in sample.labels.items():
                if key in ("severity", "collector"):
                    assert value in ALLOWED_LABEL_VALUES


# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------

def test_import_mlops_does_not_import_prometheus_client():
    exec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exec")
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import mlops; "
         "assert 'prometheus_client' not in sys.modules, 'prometheus_client leaked via import mlops'"],
        cwd=exec_dir,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_incidents_open_gauge_includes_the_most_severe_p0_bucket():
    """The real Severity enum has P0 (serving failure, null predictions) as its most
    urgent level, and OpsRepo already counts four buckets. The original gauge only
    exported P1-P3, so the incidents an on-call person most needs to see were the
    ones the metric could not show."""
    registry = CollectorRegistry()
    register_domain_metrics(registry, incidents=FakeIncidents({"P0": 3, "P1": 0, "P2": 1, "P3": 0}))
    text = scrape(registry)
    assert metric_value(text, "mlops_incidents_open", {"severity": "P0"}) == 3.0
