"""
OpenTelemetry-style observability (Claims C25, C26, C27, C28).

Requirement R3.1 packages the four observability surfaces of the MLOps
platform into a single in-process module -- no external collector, no real
Prometheus client library and no live Grafana instance are required:

* C25 -- trace emission across pipeline stages. ``Tracer``/``Span`` record
  entry/exit of every pipeline stage, gate evaluation start/end and
  promotion/rollback decisions, each trace keyed by request ID with an actor
  and a duration. Source suite: ``tests/R3_1_S1.py``.
* C26 -- structured JSON logs with no secrets. ``Logger`` emits one JSON
  object per event (request_id, timestamp, actor, action, resource, result,
  and error only on failure) and redacts secret-like resource fields.
  Source suite: ``tests/R3_1_S2.py``.
* C27 -- Prometheus metrics export. ``Counter``/``Gauge``/
  ``MetricsRegistry``/``PrometheusExporter`` track the six required pipeline
  metrics and render them in the Prometheus text exposition format.
  Source suite: ``tests/R3_1_S3.py``.
* C28 -- Grafana dashboard schema validation. ``DashboardValidator`` gates a
  self-documenting dashboard (titles, descriptions, targets, thresholds,
  axis labels/units) and reports which required metrics it references.
  Source suite: ``tests/R3_1_S4.py``.

The inline reference implementations in those suites are lifted here with the
same public names, signatures, default arguments and exception message
substrings, so behaviour matches the frozen specs. ``default_clock`` is the
production default (``time.monotonic``); ``FakeClock`` is exported for
deterministic tests. Standard library only.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def default_clock() -> float:
    """Production clock: monotonic seconds, suitable for span durations."""
    return time.monotonic()


class FakeClock:
    """Deterministic monotonic clock: each call advances by a fixed step.

    Test/support helper exported for parity with ``tests/R3_1_S1.py``.
    """

    def __init__(self, start: float = 0.0, step: float = 0.01):
        self._t = start
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


class RequestID:
    """Helper for minting the request IDs every trace and log entry carries."""

    @staticmethod
    def new() -> str:
        """Return a fresh, collision-resistant request ID."""
        return uuid.uuid4().hex


@dataclass
class Span:
    """One traced pipeline stage, gate evaluation or promotion decision."""

    stage: str
    request_id: str
    actor: str
    start_ts: float
    end_ts: Optional[float] = None
    outcome: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Elapsed milliseconds; raises if the span was never ended."""
        if self.end_ts is None:
            raise ValueError(f"span for stage '{self.stage}' was never ended")
        return (self.end_ts - self.start_ts) * 1000.0


class Tracer:
    """In-process tracer; no external collector required.

    Spans are grouped by ``request_id`` so a full pipeline run can be traced
    back. Serves C25 / ``tests/R3_1_S1.py``.
    """

    def __init__(self, clock=None):
        self._clock = clock if clock is not None else default_clock
        self._spans: Dict[str, List[Span]] = {}

    def start_span(self, stage: str, request_id: str, actor: str) -> Span:
        """Start and record a span for ``stage`` under ``request_id``."""
        if not request_id:
            raise ValueError("request_id is required to start a span")
        span = Span(
            stage=stage,
            request_id=request_id,
            actor=actor,
            start_ts=self._clock(),
        )
        self._spans.setdefault(request_id, []).append(span)
        return span

    def end_span(self, span: Span, outcome: str = "ok") -> Span:
        """Stamp ``span`` with its exit time and outcome."""
        span.end_ts = self._clock()
        span.outcome = outcome
        return span

    def get_trace(self, request_id: str) -> List[Span]:
        """Return a copy of the spans recorded for ``request_id`` in order."""
        return list(self._spans.get(request_id, []))


class Logger:
    """Structured JSON logger. Serves C26 / ``tests/R3_1_S2.py``.

    Writes one JSON object per event to an injectable sink (a list here
    stands in for stdout/file in production) and redacts secret-like fields.
    """

    SECRET_FIELD_PATTERNS = ("api_key", "password", "token", "secret", "credential")

    def __init__(self, sink=None):
        self.sink = sink if sink is not None else []

    def _redact(self, resource):
        """Return ``resource`` with secret-like dict values masked."""
        if not isinstance(resource, dict):
            return resource
        redacted = {}
        for key, value in resource.items():
            if any(
                pattern in key.lower() for pattern in self.SECRET_FIELD_PATTERNS
            ):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = value
        return redacted

    def log_event(
        self, request_id, actor, action, resource, result, error=None
    ) -> dict:
        """Build, append and return one structured JSON log entry."""
        entry = {
            "request_id": request_id,
            "timestamp": now_iso(),
            "actor": actor,
            "action": action,
            "resource": self._redact(resource),
            "result": result,
        }
        if error is not None:
            entry["error"] = str(error)
        self.sink.append(json.dumps(entry))
        return entry


class Counter:
    """Monotonically accumulating metric (Prometheus ``counter``)."""

    def __init__(self, name, help_text=""):
        self.name = name
        self.help_text = help_text
        self.value = 0

    def inc(self, amount=1):
        """Add ``amount`` to the counter's accumulated value."""
        self.value += amount


class Gauge:
    """Point-in-time metric (Prometheus ``gauge``)."""

    def __init__(self, name, help_text=""):
        self.name = name
        self.help_text = help_text
        self.value = 0.0

    def set(self, value):
        """Overwrite the gauge with the latest observed ``value``."""
        self.value = value


REQUIRED_METRICS = [
    "pipeline_duration",
    "gate_latency",
    "model_size",
    "promotion_time",
    "rollback_time",
    "canary_p95_latency",
]


class MetricsRegistry:
    """Instance-level registry of counters and gauges (C27).

    There is deliberately no module-global state, so two registries never
    interfere with each other.
    """

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}

    def counter(self, name, help_text="") -> Counter:
        """Get or create the counter named ``name``."""
        if name not in self._counters:
            self._counters[name] = Counter(name, help_text)
        return self._counters[name]

    def gauge(self, name, help_text="") -> Gauge:
        """Get or create the gauge named ``name``."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, help_text)
        return self._gauges[name]


class PrometheusExporter:
    """Render a registry in the Prometheus text exposition format (C27).

    Emits gauges first, then counters, each as ``# HELP`` + ``# TYPE`` +
    ``name value``. An empty registry exports the empty string.
    """

    def __init__(self, registry: MetricsRegistry):
        self.registry = registry

    def export(self) -> str:
        """Return the registry rendered as Prometheus text."""
        lines = []
        for gauge in self.registry._gauges.values():
            lines.append(f"# HELP {gauge.name} {gauge.help_text}")
            lines.append(f"# TYPE {gauge.name} gauge")
            lines.append(f"{gauge.name} {gauge.value}")
        for counter in self.registry._counters.values():
            lines.append(f"# HELP {counter.name} {counter.help_text}")
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(f"{counter.name} {counter.value}")
        return "\n".join(lines) + ("\n" if lines else "")


class DashboardValidationError(ValueError):
    """Raised when a Grafana dashboard fails schema validation (C28)."""


class DashboardValidator:
    """Validate a Grafana JSON dashboard's self-documenting schema (C28).

    Serves ``tests/R3_1_S4.py``: every panel must carry a title, a
    description, at least one target with an ``expr``, non-empty alert
    thresholds and either an axis label or a field-config unit.
    """

    REQUIRED_METRIC_REFS = {
        "pipeline_duration",
        "gate_latency",
        "promotion_time",
        "rollback_time",
        "canary_p95_latency",
    }

    def validate(self, dashboard: dict) -> bool:
        """Return True for a valid dashboard, else raise."""
        if not dashboard.get("title"):
            raise DashboardValidationError("dashboard missing title")
        panels = dashboard.get("panels")
        if not panels:
            raise DashboardValidationError("dashboard has no panels")
        for panel in panels:
            title = panel.get("title")
            if not title:
                raise DashboardValidationError("panel missing title")
            if not panel.get("description"):
                raise DashboardValidationError(
                    f"panel '{title}' missing description"
                )
            targets = panel.get("targets")
            if not targets or not any(t.get("expr") for t in targets):
                raise DashboardValidationError(f"panel '{title}' missing metric expr")
            if not panel.get("thresholds"):
                raise DashboardValidationError(
                    f"panel '{title}' missing alert thresholds"
                )
            has_axis_label = bool(panel.get("axis_label"))
            has_unit = bool(
                panel.get("fieldConfig", {}).get("defaults", {}).get("unit")
            )
            if not has_axis_label and not has_unit:
                raise DashboardValidationError(
                    f"panel '{title}' missing axis label/unit"
                )
        return True

    def referenced_metrics(self, dashboard: dict) -> set:
        """Return the required metric names referenced by panel target exprs."""
        names = set()
        for panel in dashboard.get("panels", []):
            for target in panel.get("targets", []):
                expr = target.get("expr", "")
                for metric in self.REQUIRED_METRIC_REFS:
                    if metric in expr:
                        names.add(metric)
        return names


__all__ = [
    "now_iso",
    "default_clock",
    "FakeClock",
    "RequestID",
    "Span",
    "Tracer",
    "Logger",
    "Counter",
    "Gauge",
    "REQUIRED_METRICS",
    "MetricsRegistry",
    "PrometheusExporter",
    "DashboardValidationError",
    "DashboardValidator",
]
