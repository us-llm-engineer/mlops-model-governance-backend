"""FastAPI application for MLOps control plane (S3.3)."""

import hashlib
import json
import math
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import AfterValidator, BaseModel, Field, field_validator, StrictBool, StrictFloat, StrictInt, StrictStr, ValidationError
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from mlops.audit import ChainedAuditStore
from mlops.storage_sqlite import DurableAuditStore
from mlops.config_lint import lint_documents, lint_score, lint_context
from mlops.drift import DriftMonitor
from mlops.kernel import (
    Clock,
    Conflict,
    IntegrityError,
    NotFound,
    PolicyDenied,
    SystemClock,
    ValidationFailed,
)
from mlops.lineage import LineageGraph
from mlops.model_stages import ModelRegistry, Stage
from mlops.policy_engine import PolicyBundle, Rule
from mlops.policy_store import PolicyStore
from mlops.svc.deps import Principal, get_principal, require_role
from mlops.svc.errors import install_error_handlers
from mlops.svc.idempotency import IdempotencyStore
from mlops.svc.settings import Settings


class AppState:
    """Application state container."""

    def __init__(
        self,
        settings: Settings,
        audit,
        registry: ModelRegistry,
        policy_store: PolicyStore,
        clock: Clock,
        repo=None,
        idempotency_store: Optional[IdempotencyStore] = None,
        metrics_registry: Optional[CollectorRegistry] = None,
        drift_monitors: Optional[dict] = None,
        dataset_registry=None,
        mlflow_bridge=None,
        drift_frame_reports: Optional[dict] = None,
        incidents_mgr=None,
        ops_repo=None,
    ):
        self.settings = settings
        self.audit = audit
        self.registry = registry
        self.policy_store = policy_store
        self.clock = clock
        self.repo = repo
        self.idempotency_store = idempotency_store
        self.metrics_registry = metrics_registry
        self.dataset_registry = dataset_registry
        # Opt-in, best-effort MLflow mirror (see settings.mlflow_tracking_uri). None
        # unless explicitly configured; a mirror failure never affects the real
        # registration/transition, which stays authoritative regardless.
        self.mlflow_bridge = mlflow_bridge
        # Populated in place by workers.drift_evaluation_tick (see build_default_wiring);
        # a status endpoint or the demo driver can read it, never required for it to exist.
        self.drift_frame_reports = drift_frame_reports if drift_frame_reports is not None else {}
        # The same IncidentManager incidents_extension's router and the incident-
        # escalation worker use, exposed here so callers (a status endpoint, the
        # demo driver) never need to construct a second, disconnected one.
        self.incidents_mgr = incidents_mgr
        # OpsRepo (drift-baseline/policy-version/incident CRUD) -- the same
        # instance the drift-evaluation worker closes over, exposed for symmetry
        # with state.repo (SqlRepo) and so callers never build a second one.
        self.ops_repo = ops_repo
        # In-memory lineage of model version records (not yet persisted across
        # restarts -- storage_sqlite.save_lineage/load_lineage would need their
        # own DurableDocs wiring, out of scope this round). Real within a running
        # service: every register/validate/transition adds a real node, and
        # consecutive records for the same (model_id, version_id) are linked by
        # a real edge, which is what /v1/lineage/*/blast-radius queries.
        self.lineage_graph = LineageGraph()
        self.lineage_node_by_version: dict = {}
        # Named DriftMonitor instances that persist across /v1/drift/check calls,
        # so a periodic worker (workers.drift_evaluation_tick) can re-evaluate them
        # on a schedule instead of only the one-shot result returned per request.
        self.drift_monitors = drift_monitors if drift_monitors is not None else {}


# Private exception for validation errors
class _Unprocessable(Exception):
    """Raised when request body validation fails."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# Private exception for body size limit
class _TooLarge(Exception):
    """Raised when request body exceeds size limit."""
    pass


class _UnsupportedMedia(Exception):
    """Raised when Content-Type is not application/json."""
    pass


def _no_control_chars(v: str) -> str:
    for ch in v:
        o = ord(ch)
        if o < 0x20 or o == 0x7F:
            raise ValueError("control characters not allowed")
    return v


Ident = Annotated[StrictStr, Field(min_length=1, max_length=256), AfterValidator(_no_control_chars)]


# Request models with strict validation
class RegisterModelRequest(BaseModel):
    model_id: Ident
    version_id: Ident
    artifact_hash: StrictStr = Field(..., pattern=r"^[a-f0-9]{64}$")
    dataset_version: Optional[Ident] = None

    model_config = {"extra": "forbid"}


class RegisterDatasetRequest(BaseModel):
    name: Ident
    content_hash: StrictStr = Field(..., pattern=r"^[a-f0-9]{64}$")
    rows: StrictInt = Field(..., ge=0)
    dataset_schema: dict = Field(...)

    model_config = {"extra": "forbid"}


class ValidateModelRequest(BaseModel):
    evidence: dict

    model_config = {"extra": "forbid"}


class TransitionModelRequest(BaseModel):
    to_stage: Literal["staging", "production", "archived"]
    context: Optional[dict] = None

    model_config = {"extra": "forbid"}


class RollbackRequest(BaseModel):
    model_config = {"extra": "forbid"}


class PublishPolicyRequest(BaseModel):
    name: Ident
    version: StrictInt = Field(..., ge=1)
    rules: list = Field(default_factory=list, max_length=200)
    note: Optional[Ident] = None

    model_config = {"extra": "forbid"}

    @field_validator("rules")
    @classmethod
    def _check_rules(cls, rules):
        for r in rules:
            if isinstance(r, dict):
                for k in ("id", "rule_id", "kind", "name"):
                    v = r.get(k)
                    if isinstance(v, str):
                        if not (1 <= len(v) <= 256):
                            raise ValueError("invalid rule identifier length")
                        _no_control_chars(v)
        return rules


class ActivatePolicyRequest(BaseModel):
    human_approved_by: Optional[Ident] = None

    model_config = {"extra": "forbid"}


class PolicyDecideRequest(BaseModel):
    action: Ident
    context: dict = Field(default_factory=dict)

    model_config = {"extra": "forbid", "allow_inf_nan": False}


class LintRequest(BaseModel):
    documents: list = Field(default_factory=list, max_length=5000)
    image_allowlist: Optional[list] = Field(default=None, max_length=5000)

    model_config = {"extra": "forbid"}


_Sample = Annotated[StrictFloat, Field(allow_inf_nan=False)] | StrictInt


class DriftCheckRequest(BaseModel):
    reference: list[_Sample] = Field(..., max_length=100000)
    window: list[_Sample] = Field(..., max_length=100000)
    # Optional: when given, the built monitor is kept in state.drift_monitors under
    # this name so the periodic drift_evaluation_tick worker can re-check it later
    # against fresh observations, instead of only the one-shot result below.
    name: Optional[StrictStr] = Field(default=None, min_length=1, max_length=256)

    model_config = {"extra": "forbid", "allow_inf_nan": False}


def _parse(model_cls: type, raw: dict) -> Any:
    """Parse and validate a request body using a Pydantic model.

    Raises _Unprocessable on validation error with field names only.
    """
    try:
        return model_cls.model_validate(raw)
    except ValidationError as e:
        # Extract field names from errors
        fields = set()
        for error in e.errors():
            loc = error.get("loc", ())
            if loc:
                field_name = str(loc[0])
                fields.add(field_name)
        field_list = ", ".join(sorted(fields)) if fields else "unknown"
        raise _Unprocessable(f"validation error in fields: {field_list}")


def create_app(
    settings: Settings,
    *,
    audit=None,
    registry=None,
    policy_store=None,
    clock=None,
    repo=None,
    extensions=None,
    workers: Optional[list] = None,
    drift_monitors: Optional[dict] = None,
    dataset_registry=None,
    mlflow_bridge=None,
    drift_frame_reports: Optional[dict] = None,
    incidents_mgr=None,
    ops_repo=None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    # Validate extensions parameter
    if extensions is not None:
        if not isinstance(extensions, (list, tuple)):
            raise ValidationFailed("extensions must be None or a list/tuple of callables")
        for ext in extensions:
            if not callable(ext):
                raise ValidationFailed("extensions must be None or a list/tuple of callables")
    # Set defaults
    if audit is None:
        secret = settings.audit_secret.get_secret_value().encode()
        if settings.db_path and settings.db_path != ":memory:":
            audit = DurableAuditStore(settings.db_path, secret)
        else:
            audit = ChainedAuditStore(secret)

    if clock is None:
        clock = SystemClock()

    if registry is None:
        # Only the default-construction path also defaults dataset_registry: a
        # caller supplying their own `registry` is responsible for whatever
        # `datasets=` it was built with (matches the existing repo=/policy_store=
        # override pattern -- overrides are trusted as-is, never second-guessed).
        if dataset_registry is None:
            from mlops.dataset_versions import DatasetRegistry

            dataset_registry = DatasetRegistry(audit=audit)
        registry = ModelRegistry(audit=audit, datasets=dataset_registry)

    if policy_store is None:
        policy_store = PolicyStore(audit=audit, clock=clock)

    if mlflow_bridge is None and settings.mlflow_tracking_uri:
        from mlops.interop.mlflow_bridge import MlflowBridge

        mlflow_bridge = MlflowBridge(settings.mlflow_tracking_uri)

    # Create idempotency store
    idempotency_store = IdempotencyStore(clock, settings.idempotency_ttl_s)

    # Create metrics registry
    metrics_registry = CollectorRegistry()
    http_requests_counter = Counter(
        "mlops_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
        registry=metrics_registry,
    )
    http_request_histogram = Histogram(
        "mlops_http_request_seconds",
        "HTTP request duration in seconds",
        ["path"],
        registry=metrics_registry,
    )
    audit_length_gauge = Gauge(
        "mlops_audit_length",
        "Audit log length",
        registry=metrics_registry,
    )

    # Create the FastAPI app
    docs_url = "/docs" if settings.env_name == "dev" else None
    redoc_url = "/redoc" if settings.env_name == "dev" else None
    is_prod = settings.env_name == "prod"

    # Define lifespan for worker management
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if workers:
            import anyio
            from mlops.svc.workers import WorkerRegistry
            async with anyio.create_task_group() as tg:
                registry = WorkerRegistry()
                for worker in workers:
                    registry.add(worker)
                registry.start_all(tg)
                yield
                registry.stop_all()
        else:
            yield

    app = FastAPI(docs_url=docs_url, redoc_url=redoc_url, openapi_url=None if is_prod else "/openapi.json", lifespan=lifespan)

    # Create app state
    state = AppState(
        settings=settings,
        audit=audit,
        registry=registry,
        policy_store=policy_store,
        clock=clock,
        repo=repo,
        idempotency_store=idempotency_store,
        metrics_registry=metrics_registry,
        drift_monitors=drift_monitors,
        dataset_registry=dataset_registry,
        mlflow_bridge=mlflow_bridge,
        drift_frame_reports=drift_frame_reports,
        incidents_mgr=incidents_mgr,
        ops_repo=ops_repo,
    )

    # Store state in app
    app.state.mlops = state

    # Override get_state dependency
    from mlops.svc.deps import get_state

    def get_app_state():
        return app.state.mlops

    app.dependency_overrides[get_state] = get_app_state

    # Exception handler for validation errors
    async def unprocessable_handler(request: Request, exc: _Unprocessable):
        # Check if this is an idempotency conflict
        if exc.message == "idempotency_conflict":
            return JSONResponse(
                status_code=422,
                content={"error": {"code": "idempotency_conflict", "message": "idempotency conflict"}},
            )
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation", "message": exc.message}},
        )

    # Exception handler for body size errors
    async def too_large_handler(request: Request, exc: _TooLarge):
        return JSONResponse(
            status_code=413,
            content={"error": {"code": "payload_too_large", "message": "request body too large"}},
        )

    async def unsupported_handler(request: Request, exc: _UnsupportedMedia):
        return JSONResponse(
            status_code=415,
            content={"error": {"code": "unsupported_media_type", "message": "Content-Type must be application/json"}},
        )

    app.add_exception_handler(_UnsupportedMedia, unsupported_handler)
    app.add_exception_handler(_Unprocessable, unprocessable_handler)
    app.add_exception_handler(_TooLarge, too_large_handler)

    # Install error handlers
    install_error_handlers(app)

    # Middleware for request metrics (no body handling: auth runs first)
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Record request counters with bounded label cardinality."""
        response = await call_next(request)

        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        http_requests_counter.labels(
            method=request.method,
            path=path,
            status=response.status_code,
        ).inc()

        audit_length_gauge.set(state.audit.head()["length"])

        return response

    # Middleware for request timing
    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        """Record request timing metrics."""
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        http_request_histogram.labels(path=path).observe(duration)

        return response

    def _json_problem(data, limit: int = 64) -> Optional[str]:
        """Iterative walk: returns 'depth' / 'nonfinite' for the first problem, else None."""
        stack = [(data, 1)]
        while stack:
            node, depth = stack.pop()
            if isinstance(node, dict):
                children = node.values()
            elif isinstance(node, list):
                children = node
            elif isinstance(node, float):
                if not math.isfinite(node):
                    return "nonfinite"
                continue
            else:
                continue
            if depth > limit:
                return "depth"
            for c in children:
                if isinstance(c, (dict, list, float)):
                    stack.append((c, depth + 1))
        return None

    async def parse_json_body(request: Request) -> dict:
        """Check content-type, size, then parse JSON (runs after auth dependencies)."""
        content_type = request.headers.get("content-type", "").lower()
        if not content_type.startswith("application/json"):
            raise _UnsupportedMedia()
        max_bytes = state.settings.max_body_bytes
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_bytes:
                    raise _TooLarge()
            except (ValueError, TypeError):
                pass

        chunks = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes:
                raise _TooLarge()
            chunks.append(chunk)
        body_bytes = b"".join(chunks)

        if not body_bytes:
            return {}

        def reject_constant(x):
            raise ValueError("invalid constant")

        try:
            data = json.loads(body_bytes.decode("utf-8"), parse_constant=reject_constant)
        except RecursionError:
            raise _Unprocessable("Body nesting too deep")
        except (ValueError, UnicodeDecodeError):
            raise _Unprocessable("Body is not valid JSON")

        if not isinstance(data, dict):
            raise _Unprocessable("Body must be a JSON object")
        problem = _json_problem(data)
        if problem == "depth":
            raise _Unprocessable("Body nesting too deep")
        if problem == "nonfinite":
            raise _Unprocessable("Body contains a non-finite number")
        return data

    if is_prod:
        @app.get("/openapi.json", include_in_schema=False)
        def openapi_json(principal: Annotated[Principal, Depends(get_principal)]):
            return JSONResponse(app.openapi())

    # Health endpoint (no auth)
    @app.get("/healthz")
    def healthz():
        """Health check endpoint."""
        return {
            "status": "ok",
            "env": state.settings.env_name,
        }

    # Metrics endpoint
    @app.get("/metrics")
    def metrics(request: Request):
        """Prometheus metrics endpoint.

        Requires a valid Bearer token by default, same as every other route (unauthenticated
        access returns 401, matching the pre-existing tested behavior). If
        `settings.metrics_public` is set (default False), a request with NO Authorization
        header at all is allowed through anonymously -- for a real Prometheus scrape config
        (e.g. a Helm-managed kube-prometheus-stack ServiceMonitor) that has no bearer token
        wired in. A request that DOES send a header is still validated normally either way,
        so a bad token is never silently accepted.
        """
        auth_header = request.headers.get("authorization")
        if auth_header or not state.settings.metrics_public:
            principal = get_principal(auth_header, state)
            if principal.role not in ("viewer", "operator", "admin"):
                raise HTTPException(status_code=403)
        text = generate_latest(state.metrics_registry).decode('utf-8')
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------------
    # Shared idempotency protocol for every keyed POST route
    # ------------------------------------------------------------------
    def _check_key(request: Request) -> Optional[str]:
        key = request.headers.get("idempotency-key")
        if key:
            try:
                idempotency_store._validate_key(key)
            except ValidationFailed:
                raise ValidationFailed("Invalid idempotency key")
        return key or None

    async def _keyed(request, principal, path, body, work, ok_status=200):
        """begin -> run work in threadpool -> finish (2xx only) / abort (anything else)."""
        key = _check_key(request)
        if key:
            body_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            status, cached = idempotency_store.begin(key, principal.name, "POST", path, body_hash)
            if status == "replay":
                return JSONResponse(status_code=cached[0], content=cached[1],
                                    headers={"Idempotent-Replayed": "true"})
            if status == "conflict":
                raise _Unprocessable("idempotency_conflict")
            if status == "in_flight":
                return JSONResponse(
                    status_code=409,
                    content={"error": {"code": "idempotency_in_flight",
                                       "message": "a request with this idempotency key is in progress"}},
                    headers={"Retry-After": "1"},
                )
        finished = False
        try:
            data = await run_in_threadpool(work)
            if key:
                idempotency_store.finish(key, principal.name, "POST", path, body_hash, ok_status, data)
                finished = True
            return JSONResponse(status_code=ok_status, content=data)
        finally:
            if key and not finished:
                idempotency_store.abort(key, principal.name, "POST", path, body_hash)

    def _mirror_record(record):
        # Real lineage node per record, linked to this (model_id, version_id)'s
        # previous node (if any) by a real edge -- what /v1/lineage/*/blast-radius
        # and /v1/lineage/*/ancestors below actually query.
        key = (record.model_id, record.version_id)
        prev_node_id = state.lineage_node_by_version.get(key)
        node_id = state.lineage_graph.add_node(
            "model_version",
            {"model_id": record.model_id, "version_id": record.version_id, "stage": record.stage.value},
            str(time.time()),
        )
        if prev_node_id is not None:
            state.lineage_graph.add_edge(prev_node_id, node_id, edge_type="transition")
        state.lineage_node_by_version[key] = node_id

        if state.mlflow_bridge is not None:
            # Best-effort, opt-in mirror: never blocks or changes the real
            # registration/transition, which stays authoritative. version_id
            # here is an arbitrary operator-chosen string (e.g. "v1"), while
            # MlflowBridge.mirror_stage_transition requires an ASCII-digit
            # version (it maps 1:1 onto MLflow's own numeric model versions) --
            # skip that specific call rather than fail on the common case.
            try:
                from mlops.experiments import LogEntry

                entries = [LogEntry("param", "artifact_hash", record.artifact_hash, None)]
                if record.dataset_version is not None:
                    entries.append(LogEntry("param", "dataset_version", record.dataset_version, None))
                state.mlflow_bridge.mirror_experiment_run(
                    entries,
                    run_id=f"{record.model_id}:{record.version_id}",
                    experiment_name=record.model_id,
                )
                if record.version_id.isdigit():
                    state.mlflow_bridge.mirror_stage_transition(
                        record.model_id, record.version_id, record.stage
                    )
            except Exception:
                pass

        if state.repo:
            ts = time.time()
            state.repo.upsert_model({
                "model_id": record.model_id,
                "version_id": record.version_id,
                "artifact_hash": record.artifact_hash,
                "dataset_version": record.dataset_version,
                "stage": record.stage.value,
                "validated": record.validated,
                "previous_production": record.previous_production,
                "created_ts": ts,
                "updated_ts": ts,
            })
        return {
            "model_id": record.model_id,
            "version_id": record.version_id,
            "artifact_hash": record.artifact_hash,
            "dataset_version": record.dataset_version,
            "stage": record.stage.value,
            "validated": record.validated,
            "previous_production": record.previous_production,
        }

    # Model endpoints
    @app.post("/v1/models")
    async def register_model(
        request: Request,
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Register a new model version."""
        _check_key(request)
        body = await parse_json_body(request)
        req = _parse(RegisterModelRequest, body)

        def work():
            return _mirror_record(state.registry.register(
                actor=principal.name,
                model_id=req.model_id,
                version_id=req.version_id,
                artifact_hash=req.artifact_hash,
                dataset_version=req.dataset_version,
            ))

        return await _keyed(request, principal, "/v1/models", body, work, 201)

    # Dataset endpoint: registers into the same DatasetRegistry instance
    # ModelRegistry.register()'s dataset_version= check validates against.
    @app.post("/v1/datasets")
    async def register_dataset(
        request: Request,
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Register a new dataset version."""
        if state.dataset_registry is None:
            raise _Unprocessable("dataset registry not configured")
        _check_key(request)
        body = await parse_json_body(request)
        req = _parse(RegisterDatasetRequest, body)

        def work():
            version_id = state.dataset_registry.register(
                actor=principal.name,
                name=req.name,
                content_hash=req.content_hash,
                rows=req.rows,
                schema=req.dataset_schema,
            )
            return {"version_id": version_id, "name": req.name}

        return await _keyed(request, principal, "/v1/datasets", body, work, 201)

    @app.post("/v1/models/{model_id}/versions/{version_id}/validate")
    async def validate_model(
        request: Request,
        model_id: str,
        version_id: str,
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Validate a model version."""
        _check_key(request)
        body = await parse_json_body(request)
        req = _parse(ValidateModelRequest, body)

        def work():
            return _mirror_record(state.registry.validate(
                actor=principal.name,
                model_id=model_id,
                version_id=version_id,
                evidence=req.evidence,
            ))

        return await _keyed(request, principal, f"/v1/models/{model_id}/versions/{version_id}/validate",
                            body, work)

    @app.post("/v1/models/{model_id}/versions/{version_id}/transition")
    async def transition_model(
        request: Request,
        model_id: str,
        version_id: str,
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
    ):
        """Transition a model version to a new stage."""
        _check_key(request)
        body = await parse_json_body(request)
        req = _parse(TransitionModelRequest, body)

        if req.to_stage == "production" and principal.role != "admin":
            raise PolicyDenied("Only admin can transition to production")

        to_stage_enum = {
            "staging": Stage.STAGING,
            "production": Stage.PRODUCTION,
            "archived": Stage.ARCHIVED,
        }[req.to_stage]

        def work():
            if principal.role != "admin":
                for rec in state.registry.history(model_id):
                    if rec.version_id == version_id and rec.stage == Stage.PRODUCTION:
                        raise PolicyDenied("Only admin can move a model out of production")
            return _mirror_record(state.registry.transition(
                actor=principal.name,
                model_id=model_id,
                version_id=version_id,
                to_stage=to_stage_enum,
                context=req.context,
            ))

        return await _keyed(request, principal, f"/v1/models/{model_id}/versions/{version_id}/transition",
                            body, work)

    @app.post("/v1/models/{model_id}/rollback")
    async def rollback_model(
        request: Request,
        model_id: str,
        principal: Annotated[Principal, Depends(require_role("admin"))],
    ):
        """Rollback to previous production version."""
        _check_key(request)
        body = await parse_json_body(request)
        _parse(RollbackRequest, body)

        def work():
            return _mirror_record(state.registry.rollback(actor=principal.name, model_id=model_id))

        return await _keyed(request, principal, f"/v1/models/{model_id}/rollback", body, work)

    @app.get("/v1/models/{model_id}")
    def get_model(
        model_id: str,
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
    ):
        """Get model version history."""
        if state.repo:
            records = state.repo.get_model(model_id)
            history = []
            for rec in records:
                history.append({
                    "model_id": rec["model_id"],
                    "version_id": rec["version_id"],
                    "artifact_hash": rec["artifact_hash"],
                    "dataset_version": rec.get("dataset_version"),
                    "stage": rec["stage"],
                    "validated": rec["validated"],
                    "previous_production": rec.get("previous_production"),
                })
            return history
        else:
            registry_history = state.registry.history(model_id)
            history = []
            for record in registry_history:
                history.append({
                    "model_id": record.model_id,
                    "version_id": record.version_id,
                    "artifact_hash": record.artifact_hash,
                    "dataset_version": record.dataset_version,
                    "stage": record.stage.value,
                    "validated": record.validated,
                    "previous_production": record.previous_production,
                })
            return history

    # Lineage endpoints: query the real in-memory LineageGraph _mirror_record
    # populates on every register/validate/transition.
    @app.get("/v1/lineage/{model_id}/{version_id}/blast-radius")
    def lineage_blast_radius(
        model_id: str,
        version_id: str,
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
    ):
        """Nodes that would be affected by a change to this model version."""
        from mlops.ext.lineage_graph import blast_radius

        node_id = state.lineage_node_by_version.get((model_id, version_id))
        if node_id is None:
            raise NotFound(f"No lineage recorded for {model_id}/{version_id}")
        return {"node_id": node_id, "blast_radius": sorted(blast_radius(state.lineage_graph, node_id))}

    @app.get("/v1/lineage/{model_id}/{version_id}/ancestors")
    def lineage_ancestors(
        model_id: str,
        version_id: str,
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
    ):
        """Nodes this model version's current state descends from."""
        from mlops.ext.lineage_graph import ancestors

        node_id = state.lineage_node_by_version.get((model_id, version_id))
        if node_id is None:
            raise NotFound(f"No lineage recorded for {model_id}/{version_id}")
        return {"node_id": node_id, "ancestors": sorted(ancestors(state.lineage_graph, node_id))}

    # Policy endpoints
    @app.post("/v1/policy/publish")
    async def publish_policy(
        request: Request,
        principal: Annotated[Principal, Depends(require_role("admin"))],
    ):
        """Publish a new policy version."""
        _check_key(request)
        body = await parse_json_body(request)
        req = _parse(PublishPolicyRequest, body)

        def work():
            rules = []
            for r in req.rules:
                if not isinstance(r, dict):
                    raise ValidationFailed("rule must be an object")
                rules.append(Rule(
                    id=r.get("id"),
                    version=r.get("version", 1),
                    kind=r.get("kind"),
                    params=r.get("params"),
                    severity=r.get("severity", "block"),
                ))
            bundle = PolicyBundle(rules=rules, name=req.name, version=req.version)
            new_version = state.policy_store.publish(
                actor=principal.name, bundle=bundle, note=req.note or "",
            )
            return {"version": new_version, "name": req.name}

        return await _keyed(request, principal, "/v1/policy/publish", body, work, 201)

    @app.post("/v1/policy/{version}/activate")
    async def activate_policy(
        request: Request,
        version: int,
        principal: Annotated[Principal, Depends(require_role("admin"))],
    ):
        """Activate a policy version."""
        _check_key(request)
        body = await parse_json_body(request)
        req = _parse(ActivatePolicyRequest, body)

        def work():
            state.policy_store.activate(
                actor=principal.name, version=version, human_approved_by=req.human_approved_by,
            )
            return {"version": version}

        return await _keyed(request, principal, f"/v1/policy/{version}/activate", body, work)

    @app.post("/v1/policy/decide")
    def policy_decide(
        principal: Annotated[Principal, Depends(require_role("operator", "admin"))],
        body: Annotated[dict, Depends(parse_json_body)],
    ):
        """Make a policy decision."""
        req = _parse(PolicyDecideRequest, body)

        try:
            version, bundle = state.policy_store.active()
        except NotFound:
            raise PolicyDenied("No active policy")

        decision = state.policy_store.decide(
            actor=principal.name,
            action=req.action,
            context=req.context,
        )

        return JSONResponse(status_code=200, content={
            "allow": decision.allow,
            "reasons": decision.reasons,
            "version": version,
            "policy_hash": bundle.hash,
        })

    @app.get("/v1/policy/active")
    def get_active_policy(
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
    ):
        """Get active policy."""
        try:
            version, bundle = state.policy_store.active()
        except NotFound:
            raise NotFound("No active policy")

        return JSONResponse(status_code=200, content={
            "version": version,
            "name": bundle.name,
        })

    # Lint endpoint
    @app.post("/v1/lint")
    def lint(
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
        body: Annotated[dict, Depends(parse_json_body)],
    ):
        """Lint Kubernetes manifests."""
        req = _parse(LintRequest, body)

        findings = lint_documents(req.documents, image_allowlist=req.image_allowlist)
        score = lint_score(findings)
        context = lint_context(findings)

        return JSONResponse(status_code=200, content={
            "findings": [
                {
                    "rule_id": f.rule_id,
                    "category": f.category,
                    "severity": f.severity,
                    "message": f.message,
                }
                for f in findings
            ],
            "score": score,
            "context": context,
        })

    # Drift endpoint
    @app.post("/v1/drift/check")
    def drift_check(
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
        body: Annotated[dict, Depends(parse_json_body)],
    ):
        """Check for data drift."""
        req = _parse(DriftCheckRequest, body)

        if len(req.window) < 30:
            raise Conflict("Window must have at least 30 samples")

        try:
            monitor = DriftMonitor(req.reference)
            for val in req.window:
                monitor.observe(val)
            result = monitor.check()
        except Conflict:
            raise
        except (ValueError, OverflowError):
            raise _Unprocessable("invalid drift input")

        if req.name is not None:
            # Keep this monitor alive under its name so the periodic
            # drift_evaluation_tick worker can re-check it later against
            # whatever window it holds at tick time. Omitting `name` keeps
            # this endpoint exactly as it was before: a one-shot check.
            state.drift_monitors[req.name] = monitor

        return JSONResponse(status_code=200, content={
            "level": result.level,
            "psi": result.psi,
            "psi_adjusted": result.psi_adjusted,
            "ks": result.ks,
            "ks_pvalue": result.ks_pvalue,
            "n": result.n,
        })

    # Audit endpoints
    @app.get("/v1/audit/head")
    def audit_head(
        principal: Annotated[Principal, Depends(require_role("viewer", "operator", "admin"))],
    ):
        """Get audit log head."""
        head = state.audit.head()
        return JSONResponse(status_code=200, content=head)

    @app.get("/v1/audit/export")
    def audit_export(
        request: Request,
        principal: Annotated[Principal, Depends(require_role("admin"))],
        limit: Optional[int] = None,
    ):
        """Export audit log entries."""
        if limit is None:
            limit = 100

        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise _Unprocessable("limit must be between 1 and 1000")

        # Get entries from the end
        all_entries = state.audit.export()
        entries = all_entries[-limit:] if limit else all_entries

        return JSONResponse(status_code=200, content={
            "entries": entries,
            "head": state.audit.head(),
        })

    # Call extensions after all routes are registered
    if extensions is not None:
        for ext in extensions:
            ext(app, state)

    return app


def build_default_wiring(settings: Settings):
    """Construct the real extensions/workers/repo this service ships with.

    This is what a `python -m mlops.svc` / serve() run actually wires in -- kept
    as its own function (rather than inlined in serve()) so both the real
    entrypoint and a test/demo driver can build the exact same live app that
    `create_app(settings)` alone (used directly by most unit tests) never wires.

    Returns a dict of kwargs ready to splat into create_app(settings, **kwargs).
    """
    from mlops.audit import ChainVerifier
    from mlops.dataset_versions import DatasetRegistry
    from mlops.incidents import IncidentManager
    from mlops.sqldb import SqlRepo, upgrade as sqldb_upgrade
    from mlops.sqldb_ops import OpsRepo
    from mlops.svc.extensions import (
        audit_extension,
        domain_metrics_extension,
        incidents_extension,
        policy_history_extension,
        telemetry_extension,
        _IncidentCounts,
    )
    from mlops.svc.telemetry import Telemetry
    from mlops.svc.workers import PeriodicWorker, drift_evaluation_tick, incident_escalation_tick

    secret = settings.audit_secret.get_secret_value().encode()
    clock = SystemClock()
    has_real_db = bool(settings.db_path) and settings.db_path != ":memory:"

    if has_real_db:
        audit = DurableAuditStore(settings.db_path, secret)
    else:
        audit = ChainedAuditStore(secret)

    policy_store = PolicyStore(audit=audit, clock=clock)
    dataset_registry = DatasetRegistry(audit=audit)
    registry = ModelRegistry(audit=audit, datasets=dataset_registry)
    incidents_mgr = IncidentManager(audit, clock)
    telemetry = Telemetry()
    drift_monitors: dict = {}
    drift_frame_reports: dict = {}

    extensions = [
        telemetry_extension(telemetry),
        incidents_extension(incidents_mgr),
        audit_extension(lambda: ChainVerifier(secret)),
        policy_history_extension(policy_store),
        domain_metrics_extension(audit=audit, incidents=_IncidentCounts(incidents_mgr)),
    ]

    workers = [
        PeriodicWorker(
            "incident-escalation",
            settings.incident_escalation_interval_s,
            lambda: incident_escalation_tick(incidents_mgr),
        ),
    ]

    repo = None
    ops_repo = None
    if has_real_db:
        # SqlRepo (model/dataset mirror, used by state.repo in app.py's
        # _mirror_record/get_model) and OpsRepo (policy-version/incident/drift
        # baseline CRUD, used by the drift worker below) are two query
        # interfaces over the SAME Alembic-migrated tables (mlops.sqldb.Base),
        # so one migrated `sqlite:///` URL backs both -- migrate it once here.
        sql_url = f"sqlite:///{settings.db_path}"
        sqldb_upgrade(sql_url, "head")
        repo = SqlRepo(sql_url)
        baseline_repo = ops_repo = OpsRepo(sql_url)
        workers.append(
            PeriodicWorker(
                "drift-evaluation",
                settings.drift_evaluation_interval_s,
                lambda: drift_evaluation_tick(drift_monitors, baseline_repo, drift_frame_reports),
            )
        )
    # else: no real database configured (dev/:memory: mode) -- SqlRepo/OpsRepo
    # cannot be constructed against a non-persistent URL, so model/dataset
    # mirroring and drift baselines simply aren't persisted in that mode.
    # Everything else above still works.

    return {
        "audit": audit,
        "registry": registry,
        "policy_store": policy_store,
        "clock": clock,
        "repo": repo,
        "extensions": extensions,
        "workers": workers,
        "drift_monitors": drift_monitors,
        "drift_frame_reports": drift_frame_reports,
        "dataset_registry": dataset_registry,
        "incidents_mgr": incidents_mgr,
        "ops_repo": ops_repo,
    }


def serve(settings: Settings, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Build the real, fully-wired app and serve it with uvicorn (imported lazily).

    Defaults to 0.0.0.0 so a containerized deployment (e.g. behind a Helm-managed
    Service/Ingress) is reachable from outside the container without an override.
    Unlike calling create_app(settings) alone, this wires in the extensions,
    workers, and OpsRepo the service actually ships with (see build_default_wiring).
    """
    import uvicorn

    app = create_app(settings, **build_default_wiring(settings))
    uvicorn.run(app, host=host, port=port)
