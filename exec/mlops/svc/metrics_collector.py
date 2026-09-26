"""Domain metrics collection for MLOps control plane observability.

Implements Collector interface for prometheus_client, providing metrics for
audit chain, policy store, incidents, and model monitoring.
"""

from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
)
from prometheus_client.registry import CollectorRegistry

from mlops.kernel import ValidationFailed, NotFound, Conflict


class DomainMetricsCollector:
    """Collects domain metrics from pluggable providers.

    Keyword-only constructor with optional audit, policy_store, incidents,
    and models providers. Implements prometheus Collector interface.
    """

    def __init__(self, *, audit=None, policy_store=None, incidents=None, models=None):
        """Initialize collector with optional providers."""
        self.audit = audit
        self.policy_store = policy_store
        self.incidents = incidents
        self.models = models
        self._error_counts = {
            "audit": 0,
            "policy": 0,
            "incidents": 0,
            "models": 0,
        }

    def describe(self):
        """Return metric family descriptions (required for duplicate detection)."""
        families = []

        if self.audit is not None:
            families.append(
                GaugeMetricFamily(
                    "mlops_audit_chain_length",
                    "Length of audit chain",
                )
            )

        if self.policy_store is not None:
            families.append(
                GaugeMetricFamily(
                    "mlops_policy_active_version",
                    "Active policy version (0 if none)",
                )
            )
            families.append(
                GaugeMetricFamily(
                    "mlops_policy_versions",
                    "Total number of policy versions",
                )
            )

        if self.incidents is not None:
            families.append(
                GaugeMetricFamily(
                    "mlops_incidents_open",
                    "Number of open incidents by severity",
                    labels=["severity"],
                )
            )

        # Error counter is always reported if any errors occur
        families.append(
            CounterMetricFamily(
                "mlops_collector_errors",
                "Total errors during metric collection by collector",
                labels=["collector"],
            )
        )

        return families

    def collect(self):
        """Yield metric families for registered providers."""
        # Audit family
        if self.audit is not None:
            try:
                chain_length = self.audit.head()["length"]
                family = GaugeMetricFamily(
                    "mlops_audit_chain_length",
                    "Length of audit chain",
                    value=float(chain_length),
                )
                yield family
            except Exception:
                self._error_counts["audit"] += 1

        # Policy families
        if self.policy_store is not None:
            try:
                # Active version
                try:
                    active_version = self.policy_store.active()[0]
                except (NotFound, Conflict):
                    # No active policy is normal, return 0
                    active_version = 0
                family = GaugeMetricFamily(
                    "mlops_policy_active_version",
                    "Active policy version (0 if none)",
                    value=float(active_version),
                )
                yield family

                # Policy versions count
                versions_count = len(self.policy_store.history())
                family = GaugeMetricFamily(
                    "mlops_policy_versions",
                    "Total number of policy versions",
                    value=float(versions_count),
                )
                yield family
            except Exception:
                self._error_counts["policy"] += 1

        # Incidents family
        if self.incidents is not None:
            try:
                counts = self.incidents.open_counts()
                family = GaugeMetricFamily(
                    "mlops_incidents_open",
                    "Number of open incidents by severity",
                    labels=["severity"],
                )
                for severity in ["P0", "P1", "P2", "P3"]:
                    family.add_metric([severity], float(counts.get(severity, 0)))
                yield family
            except Exception:
                self._error_counts["incidents"] += 1

        # Error counter family (always present if any errors recorded)
        error_family = CounterMetricFamily(
            "mlops_collector_errors",
            "Total errors during metric collection by collector",
            labels=["collector"],
        )
        for collector_name in ["audit", "policy", "incidents", "models"]:
            error_family.add_metric(
                [collector_name],
                float(self._error_counts[collector_name]),
            )
        yield error_family


def register_domain_metrics(registry, **providers) -> DomainMetricsCollector:
    """Register domain metrics collector on a prometheus registry.

    Args:
        registry: prometheus_client.CollectorRegistry instance
        **providers: keyword arguments for DomainMetricsCollector
                    (audit, policy_store, incidents, models)

    Returns:
        DomainMetricsCollector instance registered on the registry

    Raises:
        ValidationFailed: if registry is not a CollectorRegistry
        ValueError: if collector is already registered on this registry
    """
    if not isinstance(registry, CollectorRegistry):
        raise ValidationFailed("registry must be a CollectorRegistry")

    collector = DomainMetricsCollector(**providers)
    registry.register(collector)
    return collector
