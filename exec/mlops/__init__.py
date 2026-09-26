"""
mlops -- MLOps Control Plane backend (Round 1).

A self-contained, importable backend implementing the Round 1 claims:

R1.1 -- Auth/Tenancy + Declarative Pipeline/Environment Specs
    C1 schema.py        immutable-reference validation for declarative specs
    C2 rbac.py          RBAC enforcement with a tamper-evident audit trail
    C3 secrets.py       secret reference validation and in-memory isolation
    C4 persistence.py   lossless config round-trip and lineage/audit ordering

R1.2 -- Workflow Orchestration + Artifact Registry & Lineage
    C5 state_machine.py pipeline-run state machine (Pending->Running->terminal)
    C6 artifacts.py     content-addressed, immutable artifact storage
    C7 lineage.py       content-addressed lineage DAG with tamper detection
    C8 retry.py         failure/retry semantics and per-attempt artifact versioning

R1.3 -- Platform API & CLI (Developer Self-Service)
    C9  api.py          metadata API contract, pagination and latency budget
    C10 api.py          structured JSON error handling (no stack traces)
    C11 cli.py          CLI -> API integration and fail-fast exit codes
    C12 latency.py      end-to-end latency and concurrent-operation handling

The frozen Round 1 suites (tests/R1_*.py) are self-contained; this package is
the standalone production implementation of the behaviour those suites pin.
"""

from .legacy.spec import PipelineSpec, PipelineValidator
from .rbac import AuditEvent, Principal, RBACEngine, Role
from .legacy.secrets import (
    PipelineSpecValidator,
    SecretBinding,
    SecretResolver,
    SecretStore,
)
from .legacy.persistence import (
    AuditStore as PersistenceAuditStore,
    LineageNode as PersistenceLineageNode,
    LineageTracker,
    PersistenceLayer,
    PipelineConfig,
)
from .state_machine import (
    JobSimulator,
    JobStatus,
    PipelineRun,
    PipelineRunManager,
    RunState,
    StateMachine,
)
from .legacy.artifacts import ArtifactMetadata, ArtifactStore, PipelineStageExecutor
from .lineage import (
    LineageGraph,
    LineageGraphNode,
    LineageNode as GraphLineageNode,
    PipelineLineageBuilder,
)
from .legacy.retry import ArtifactRegistry, RetryAttempt, RetryManager, RunWithRetries
from .legacy.api import (
    ApiResponse,
    ApiServer,
    ApiValidator,
    ErrorResponse,
    InMemoryDataStore,
    ModelApi,
    PipelineApi,
)
from .legacy.cli import CliApiClient, CliCommand, CliHandler, StubApiClient
from .legacy.latency import (
    ApiBackend,
    CliExecutor,
    ConcurrencyTestbed,
    LatencyMonitor,
    Operation,
)

# R2.1 -- CI/CD policy gates (C13-C16)
from .gates import AuditEntry as GateAuditEntry, DataContractGate, GateDecision, PolicyEngine
from .legacy.evaluation import (
    BaselineMetrics,
    BaselineStore,
    EvaluationMetrics,
    ModelComparator,
    ModelQualityGate,
)
# R2.2 -- environment promotion & rollback (C17-C20)
from .promotion import (
    EnvironmentManifest,
    Principal as PromotionPrincipal,
    PromotionApprover,
    PromotionAuditEvent,
    PromotionExecutor,
    PromotionRequest,
    PromotionRole,
)
from .legacy.rollback import RollbackExecutor, RollbackRequest, RollbackValidator
# R2.3 -- Kubernetes serving & canary analysis (C21-C24)
from .legacy.metrics import (
    CanaryMetrics,
    MetricsCollector,
    RequestMetric,
    StatisticalJudge,
    mann_whitney_u,
)
from .legacy.serving import (
    BlueGreenController,
    CanaryAnalyzer,
    Deployment,
    DeploymentState,
    ModelServingDeployment,
    ServiceConfigError,
    SharedService,
)

# R3.1 -- OpenTelemetry observability (C25-C28)
from .legacy.observability import (
    REQUIRED_METRICS,
    Counter,
    DashboardValidationError,
    DashboardValidator,
    FakeClock,
    Gauge,
    Logger,
    MetricsRegistry,
    PrometheusExporter,
    RequestID,
    Span,
    Tracer,
    default_clock,
)
# R3.2 -- audit & evidence export (C29-C32)
from .audit import (
    AuditEntry,
    AuditExporter,
    AuditStore,
    ChainVerifier,
    ChainedAuditStore,
    IndexedAuditStore,
    canonical_payload,
)
# R3.3 -- failure/chaos resilience (C33-C36)
from .chaos import (
    FAULT_TYPES,
    RECOVERABLE_FAULTS,
    ArtifactStoreUnavailable,
    BypassRejected,
    ChaosScenario,
    FaultInjector,
    KubernetesJobKilled,
    PipelineJourney,
    PolicyGateTimeout,
    PromotionGuard,
    RecoveryVerifier,
    ResilienceController,
    RollbackFailed,
    SharedPlatformState,
    TrainingFault,
)

# X1.1 -- feature definition registry (C37-C40)
from .features import (
    ContractViolation,
    FeatureError,
    FeaturePermissionError,
    FeatureRegistry,
    ImmutableVersionError,
    MutableReferenceError,
)
# X1.2 -- offline/online retrieval from one definition (C41-C44, C47)
from .feature_store import FeatureStore, evaluate_feature
# X1.3 -- parity checker and promotion gate (C45, C46, C48)
from .feature_parity import ParityGate, ParityReport, check_parity

# X2.1/X2.2 -- experiment tracker (C49-C54)
from .experiments import (
    BackfillConflictError,
    ExperimentError,
    ExperimentTracker,
    LogEntry,
    RunNotWritableError,
)
# X2.2/X2.3 -- experiment registry (C55-C60)
from .experiment_registry import ExperimentRegistry

# X3.1 -- Decimal-exact cost ledger & attribution (C61-C64)
from .cost_ledger import (
    CostRow,
    attribute,
    fallback_only_spend,
    tag_namespace,
    total_spend,
    unallocated_fraction,
)
# X3.2 -- budget gate & reconciliation (C65-C68)
from .cost_gate import (
    budget_gate,
    detect_double_counted,
    exclude_source,
    reconcile,
)
# X3.3 -- tenant SLO floor & burn-rate (C69-C72)
from .slo_guard import (
    BudgetAlertTracker,
    StabilityError,
    TenantReservation,
    burn_rate,
    sojourn_bound,
    stability,
)

# S1.1 -- kernel primitives (C73-C76)
from .kernel import (
    Clock,
    Event,
    EventLog,
    IntegrityError,
    ManualClock,
    MlopsError,
    NotFound,
    PolicyDenied,
    SystemClock,
    ValidationFailed,
    Conflict,
    canonical_json,
    content_id,
    iso,
    sha256_hex,
)
# S1.2 -- policy engine (C77-C78)
from .policy_engine import (
    PolicyBundle,
    PolicyDecision,
    PolicyDecisionPoint,
    Rule,
    load_policy,
)
# S1.3 -- ML test score (C79)
from .ml_test_score import (
    ITEMS,
    ScoreReport,
    evaluate_ml_test_score,
    score_to_context,
)
# S1.4 -- gate experiment (C80-C83)
from .gate_experiment import (
    PolicyArm,
    Release,
    StaticGateArm,
    UngatedArm,
    default_policy_bundle,
    format_report,
    generate_workload,
    run_experiment,
    run_replicates,
    sensitivity_sweep,
    workload_diagnostic,
)
# S1.5 -- enforcement model (C84)
from .enforcement_model import (
    EnforcementModel,
    Location,
)

# S2.1 -- floor guard with assured-first dispatch (C85-C88)
from .floor_guard import (
    AdmitResult,
    AssuredFirstDispatcher,
    FloorConfig,
    Outcome,
    Request,
    TenantFloor,
    check_preconditions,
    require_guarantee,
    simulate_guard,
    sojourn_bound as floor_sojourn_bound,
)
# S2.2 -- durable storage: audit, docs, lineage (C89-C92)
from .storage_sqlite import (
    DurableAuditStore,
    DurableDocs,
    load_lineage,
    save_lineage,
)
# S2.3 -- release control plane (C93-C96)
from .model_stages import (
    ModelRegistry,
    ModelVersionRecord,
    Stage,
)
from .dataset_versions import (
    DatasetRegistry,
    DatasetVersionRecord,
)
from .supply_chain import (
    RetentionPolicy,
    build_attestation,
    merkle_root,
    sbom_from_requirements,
    verify_attestation,
)

# S3.1 -- policy store and calibration (C97-C102)
from .policy_store import (
    PolicyStore,
    is_stricter_or_equal,
)
from .policy_calibration import (
    Calibrator,
    Feedback,
    Proposal,
    evaluate,
)
# S3.1 -- config linting (C103-C105)
from .config_lint import (
    CATEGORY_WEIGHTS,
    Finding,
    lint_context,
    lint_documents,
    lint_manifest,
    lint_score,
    load_manifests_json,
)
# S3.2 -- drift monitoring (C106-C107)
from .drift import (
    DriftMonitor,
    DriftReport,
    classify_drift,
    cohens_kappa,
    drift_context,
    js_divergence,
    kl_divergence,
    ks_statistic,
    predictive_entropy,
    psi,
    psi_from_samples,
)
# S3.2 -- incident management (C108-C109)
from .incidents import (
    CHECKLIST,
    IncidentManager,
    Severity,
    classify,
)
# S3.2 -- chaos engineering (C110-C112)
from .chaos_scenarios import (
    FAULT_TYPES,
    FaultSpec,
    Scenario,
    SimulatedCluster,
    SteadyState,
    chaos_context,
    run_scenario,
)

__version__ = "1.3.0-x3"

__all__ = [
    # R1.1
    "PipelineSpec", "PipelineValidator",
    "Role", "Principal", "AuditEvent", "RBACEngine",
    "SecretBinding", "SecretStore", "PipelineSpecValidator", "SecretResolver",
    "PipelineConfig", "PersistenceLayer", "LineageTracker", "PersistenceAuditStore",
    "PersistenceLineageNode",
    # R1.2
    "RunState", "JobStatus", "PipelineRun", "StateMachine", "JobSimulator",
    "PipelineRunManager",
    "ArtifactMetadata", "ArtifactStore", "PipelineStageExecutor",
    "LineageGraphNode", "LineageGraph", "PipelineLineageBuilder",
    "GraphLineageNode",
    "RetryAttempt", "ArtifactRegistry", "RunWithRetries", "RetryManager",
    # R1.3
    "ApiResponse", "InMemoryDataStore", "ApiServer", "ApiValidator",
    "ErrorResponse", "PipelineApi", "ModelApi",
    "CliCommand", "CliHandler", "StubApiClient", "CliApiClient",
    "Operation", "LatencyMonitor", "ApiBackend", "CliExecutor",
    "ConcurrencyTestbed",
    # R2.1
    "GateDecision", "GateAuditEntry", "DataContractGate", "PolicyEngine",
    "BaselineMetrics", "EvaluationMetrics", "BaselineStore", "ModelQualityGate",
    "ModelComparator",
    # R2.2
    "PromotionRole", "PromotionPrincipal", "PromotionRequest",
    "PromotionAuditEvent", "EnvironmentManifest", "PromotionApprover",
    "PromotionExecutor",
    "RollbackRequest", "RollbackValidator", "RollbackExecutor",
    # R2.3
    "RequestMetric", "CanaryMetrics", "MetricsCollector", "StatisticalJudge",
    "mann_whitney_u",
    "DeploymentState", "Deployment", "ServiceConfigError", "SharedService",
    "ModelServingDeployment", "BlueGreenController", "CanaryAnalyzer",
    # R3.1
    "RequestID", "Span", "Tracer", "Logger", "Counter", "Gauge",
    "REQUIRED_METRICS", "MetricsRegistry", "PrometheusExporter",
    "DashboardValidationError", "DashboardValidator",
    "FakeClock", "default_clock",
    # R3.2
    "AuditEntry", "AuditStore", "AuditExporter", "IndexedAuditStore",
    "ChainedAuditStore", "ChainVerifier", "canonical_payload",
    # R3.3
    "KubernetesJobKilled", "ArtifactStoreUnavailable", "PolicyGateTimeout",
    "RECOVERABLE_FAULTS", "FAULT_TYPES", "FaultInjector", "ResilienceController",
    "SharedPlatformState", "BypassRejected", "PromotionGuard",
    "TrainingFault", "RollbackFailed", "PipelineJourney",
    "ChaosScenario", "RecoveryVerifier",
    # X1.1
    "FeatureError", "MutableReferenceError", "ImmutableVersionError",
    "FeaturePermissionError", "ContractViolation", "FeatureRegistry",
    # X1.2
    "FeatureStore", "evaluate_feature",
    # X1.3
    "ParityReport", "check_parity", "ParityGate",
    # X2.1/X2.2
    "ExperimentError", "RunNotWritableError", "BackfillConflictError",
    "LogEntry", "ExperimentTracker",
    # X2.3
    "ExperimentRegistry",
    # X3.1
    "CostRow", "attribute", "total_spend", "unallocated_fraction",
    "fallback_only_spend", "tag_namespace",
    # X3.2
    "budget_gate", "reconcile", "detect_double_counted", "exclude_source",
    # X3.3
    "StabilityError", "TenantReservation", "sojourn_bound", "stability",
    "burn_rate", "BudgetAlertTracker",
    # S1.1
    "Clock", "Event", "EventLog", "IntegrityError", "ManualClock",
    "MlopsError", "NotFound", "PolicyDenied", "SystemClock", "ValidationFailed",
    "Conflict", "canonical_json", "content_id", "iso", "sha256_hex",
    # S1.2
    "PolicyBundle", "PolicyDecision", "PolicyDecisionPoint", "Rule", "load_policy",
    # S1.3
    "ITEMS", "ScoreReport", "evaluate_ml_test_score", "score_to_context",
    # S1.4
    "PolicyArm", "Release", "StaticGateArm", "UngatedArm",
    "default_policy_bundle", "format_report", "generate_workload",
    "run_experiment", "run_replicates", "sensitivity_sweep", "workload_diagnostic",
    # S1.5
    "EnforcementModel", "Location",
    # S2.1
    "FloorConfig", "AdmitResult", "TenantFloor", "Request", "Outcome",
    "AssuredFirstDispatcher", "check_preconditions", "require_guarantee",
    "floor_sojourn_bound", "simulate_guard",
    # S2.2
    "DurableAuditStore", "DurableDocs", "save_lineage", "load_lineage",
    # S2.3
    "ModelRegistry", "ModelVersionRecord", "Stage",
    "DatasetRegistry", "DatasetVersionRecord",
    "RetentionPolicy", "merkle_root", "build_attestation", "verify_attestation",
    "sbom_from_requirements",
    # S3.1
    "PolicyStore", "is_stricter_or_equal",
    "Calibrator", "Feedback", "Proposal", "evaluate",
    "CATEGORY_WEIGHTS", "Finding", "lint_context", "lint_documents", "lint_manifest",
    "lint_score", "load_manifests_json",
    # S3.2
    "DriftMonitor", "DriftReport", "classify_drift", "cohens_kappa", "drift_context",
    "js_divergence", "kl_divergence", "ks_statistic", "predictive_entropy", "psi",
    "psi_from_samples",
    "CHECKLIST", "IncidentManager", "Severity", "classify",
    "FAULT_TYPES", "FaultSpec", "Scenario", "SimulatedCluster", "SteadyState",
    "chaos_context", "run_scenario",
]
