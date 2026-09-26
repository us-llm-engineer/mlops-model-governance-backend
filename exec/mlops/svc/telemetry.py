"""F1.2 Telemetry service: W3C-compliant distributed tracing (contract in the API contract section 2).

Builds its own TracerProvider per instance (never touches the global provider).
Middleware extracts W3C traceparent, starts SERVER spans named by route template,
records method/route/status attributes, sets X-Trace-Id response header.
"""

from typing import Optional, Callable, Any
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.propagate import extract

from mlops import kernel


class Telemetry:
    """Telemetry service with W3C-compliant distributed tracing."""

    def __init__(self, service_name: str = "mlops-control-plane", exporter: Optional[SpanExporter] = None):
        """Initialize Telemetry with its own TracerProvider.

        Args:
            service_name: Non-empty str with no control chars, else ValidationFailed.
            exporter: Optional SpanExporter; uses InMemorySpanExporter if None.
        """
        # Validate service_name
        if not isinstance(service_name, str):
            raise kernel.ValidationFailed("service_name must be a string")
        if not service_name:
            raise kernel.ValidationFailed("service_name must be non-empty")
        if any(ord(c) < 32 for c in service_name):  # control characters
            raise kernel.ValidationFailed("service_name must not contain control characters")

        # Use InMemorySpanExporter if none provided
        if exporter is None:
            exporter = InMemorySpanExporter()

        self._exporter = exporter
        self._installed_on_app = None  # Track which app this is installed on (if any)

        # Build our own TracerProvider (never call trace.set_tracer_provider)
        resource = Resource.create({"service.name": service_name})
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(SimpleSpanProcessor(self._exporter))

    @property
    def tracer(self):
        """Return the tracer from this Telemetry's TracerProvider."""
        return self._tracer_provider.get_tracer(__name__)

    def finished_spans(self):
        """Return finished spans (only meaningful for InMemorySpanExporter, else [])."""
        if isinstance(self._exporter, InMemorySpanExporter):
            return self._exporter.get_finished_spans()
        return []

    def shutdown(self) -> None:
        """Idempotent shutdown of the tracer provider."""
        self._tracer_provider.force_flush()
        self._tracer_provider.shutdown()

    def install(self, app) -> None:
        """Install HTTP middleware for tracing.

        Raises Conflict if already installed on this app.
        """
        if self._installed_on_app is app:
            raise kernel.Conflict("Telemetry already installed on this app")

        self._installed_on_app = app

        telemetry = self

        class TelemetryMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next: Callable) -> Any:
                # Extract W3C traceparent header (garbage ignored, fresh trace created)
                ctx = extract({"traceparent": request.headers.get("traceparent", "")})

                # Best-effort pre-dispatch guess: top-level route.matches() cannot see
                # through a router mounted via app.include_router() (it matches as one
                # opaque unit with no .path of its own), so this guess is corrected
                # below once Starlette has actually resolved the leaf route.
                route_template = "unmatched"
                for route in request.app.routes:
                    match, child_scope = route.matches(request.scope)
                    if match == Match.FULL:
                        route_template = getattr(route, "path", "unmatched")
                        break

                # Create span name from METHOD and route_template
                span_name = f"{request.method} {route_template}"

                def _resolve_route_template() -> str:
                    """Read the leaf route Starlette resolved during dispatch.

                    request.scope["route"] is only populated once routing has actually
                    run (inside call_next), and it is always the true leaf APIRoute --
                    including through app.include_router() -- unlike the pre-dispatch
                    scan above.
                    """
                    resolved = request.scope.get("route")
                    if resolved is None:
                        return "unmatched"
                    return getattr(resolved, "path", route_template)

                # Start SERVER span in the extracted context
                with telemetry._tracer_provider.get_tracer(__name__).start_as_current_span(
                    span_name, context=ctx, kind=trace.SpanKind.SERVER
                ) as span:
                    # Set attributes (route_template refined after dispatch, see below)
                    span.set_attribute("http.request.method", request.method)

                    try:
                        response = await call_next(request)
                        route_template = _resolve_route_template()
                        span.update_name(f"{request.method} {route_template}")
                        span.set_attribute("http.route", route_template)
                        # Set status code attribute and check for error status
                        span.set_attribute("http.response.status_code", response.status_code)
                        if response.status_code >= 500:
                            span.set_status(trace.Status(trace.StatusCode.ERROR))
                        # Add X-Trace-Id header
                        response.headers["X-Trace-Id"] = format(span.context.trace_id, "032x")
                        return response
                    except Exception as exc:
                        # Routing happens before the endpoint runs, so scope["route"]
                        # is already resolved even when the endpoint itself raised.
                        route_template = _resolve_route_template()
                        span.update_name(f"{request.method} {route_template}")
                        span.set_attribute("http.route", route_template)
                        # Record exception as error and re-raise
                        span.set_status(trace.Status(trace.StatusCode.ERROR))
                        # Don't put exception message or details on span
                        raise

        app.add_middleware(TelemetryMiddleware)


def current_trace_id() -> Optional[str]:
    """Return the active span's trace ID as 32 lower-hex chars, or None."""
    span = trace.get_current_span()
    if span is None or not span.is_recording():
        return None
    trace_id = span.context.trace_id
    if trace_id == 0:
        return None
    return format(trace_id, "032x")


def audit_meta_with_trace(meta: Optional[dict]) -> dict:
    """Return a new dict = meta + 'trace_id' when a valid span is active.

    Never mutates the argument. Never overwrites an existing 'trace_id'.
    """
    result = dict(meta) if meta else {}

    # Only add trace_id if not already present
    if "trace_id" not in result:
        tid = current_trace_id()
        if tid is not None:
            result["trace_id"] = tid

    return result
