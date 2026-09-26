"""Extensions for FastAPI app to wire up observability, incidents, policy, and metrics."""

from mlops.incidents import IncidentManager
from mlops.kernel import ValidationFailed
from mlops.policy_store import PolicyStore
from mlops.svc.routers.incidents import make_incidents_router
from mlops.svc.routers.audit import make_audit_router
from mlops.svc.routers.policy_history import make_policy_history_router
from mlops.svc.metrics_collector import register_domain_metrics


class _IncidentCounts:
    """Adapter to provide incident counts for metrics collector.

    Reads manager._incidents to count open incidents by severity.
    """

    def __init__(self, manager: IncidentManager):
        """Initialize with an IncidentManager."""
        self.manager = manager

    def open_counts(self) -> dict[str, int]:
        """Return count of open incidents by severity (P0, P1, P2, P3).

        Returns:
            dict with keys "P0", "P1", "P2", "P3" mapping to non-negative int counts
        """
        counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0}

        # Snapshot via dict(...) first: the manager has no lock, and incidents
        # can be opened concurrently from other requests/threads while a
        # /metrics scrape is in flight. dict(d) copies through a single C-level
        # call that does not release the GIL mid-copy, so it cannot raise
        # "dictionary changed size during iteration" the way iterating
        # manager._incidents.values() directly can under concurrent writes.
        snapshot = dict(self.manager._incidents)

        for record in snapshot.values():
            if record.get("status") == "open":
                # severity is an Enum, get its name
                severity_enum = record.get("severity")
                if hasattr(severity_enum, "name"):
                    severity_name = severity_enum.name
                    if severity_name in counts:
                        counts[severity_name] += 1

        return counts


def telemetry_extension(telemetry):
    """Create an extension that installs telemetry middleware.

    Args:
        telemetry: Telemetry instance to install

    Returns:
        Callable[app, state] that installs telemetry on the app
    """
    def ext(app, state):
        telemetry.install(app)

    return ext


def incidents_extension(manager: IncidentManager):
    """Create an extension that mounts the incidents router.

    Args:
        manager: IncidentManager instance

    Returns:
        Callable[app, state] that mounts the incidents router
    """
    def ext(app, state):
        router = make_incidents_router(manager)
        app.include_router(router)

    return ext


def audit_extension(chain_verifier_cls):
    """Create an extension that mounts the audit router.

    Args:
        chain_verifier_cls: A callable (with no args) that returns a verifier with
                          verify_export(export, head=None) -> bool method.
                          Must be callable, else raises ValidationFailed.

    Returns:
        Callable[app, state] that mounts the audit router

    Raises:
        ValidationFailed if chain_verifier_cls is not callable
    """
    if not callable(chain_verifier_cls):
        raise ValidationFailed("chain_verifier_cls must be callable")

    def ext(app, state):
        router = make_audit_router(state.audit, chain_verifier_cls)
        app.include_router(router)

    return ext


def policy_history_extension(store: PolicyStore = None):
    """Create an extension that mounts the policy history router.

    Args:
        store: Optional PolicyStore instance (if None, will use app.state.mlops.policy_store)

    Returns:
        Callable[app, state] that mounts the policy history router
    """
    def ext(app, state):
        # Use provided store or get from app state
        _store = store or state.policy_store
        router = make_policy_history_router(_store)
        app.include_router(router)

    return ext


def domain_metrics_extension(**providers):
    """Create an extension that registers domain metrics collector.

    Args:
        **providers: keyword arguments for DomainMetricsCollector
                    (audit, policy_store, incidents, models)

    Returns:
        Callable[app, state] that registers metrics on app.state.mlops.metrics_registry

    Raises:
        ValidationFailed (at call time) if metrics_registry is None
    """
    def ext(app, state):
        if state.metrics_registry is None:
            raise ValidationFailed("metrics_registry must not be None")

        register_domain_metrics(state.metrics_registry, **providers)

    return ext
