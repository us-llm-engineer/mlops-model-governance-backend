# MLOps model governance backend

A small, testable backend for governing ML models: a model registry with stage transitions,
a versioned policy layer, audit hash chain, incident lifecycle, drift checks, dataset and
lineage tracking, and an optional MLflow mirror. Python 3.12, FastAPI, SQLite (Alembic
migrations), pytest.

## Try it (terminal only)

```bash
pip install -r requirements.txt
PYTHONPATH=exec python3 -m mlops.demo        # end-to-end walk through the real app, prints every response
PYTHONPATH=exec python3 -m mlops.svc --check # build the real wiring and exit
PYTHONPATH=exec python3 -m mlops.svc --port 8000   # serve
PYTHONPATH=exec python3 -m pytest tests -q   # full suite (about 9 minutes)
```

`mlops.demo` registers a model, publishes and evaluates a scoped policy, lints a manifest,
checks drift, opens and escalates an incident, verifies the audit chain, builds and
self-validates an SBOM, and queries lineage. It uses the same `build_default_wiring` that
`python -m mlops.svc` uses, with no mocks.

## Layout

| Path | What is there |
|---|---|
| `exec/mlops/svc/` | FastAPI service: `app.py` (routes and `build_default_wiring`), settings, CLI, SDK, routers, background workers |
| `exec/mlops/policy_engine.py`, `policy_store.py` | Policy layer: rule evaluation, versioned bundles, loosening guard, decision log, replay |
| `exec/mlops/model_stages.py`, `incidents.py`, `drift.py`, `lineage.py`, `dataset_versions.py` | Registry stages, incidents, drift monitors, lineage graph, dataset registry |
| `exec/mlops/audit.py`, `sqldb*.py`, `migrations/` | Audit chain and SQLite persistence |
| `exec/mlops/interop/` | MLflow bridge (opt-in mirror) |
| `exec/mlops/legacy/` | Project 1 modules kept as a stdlib-only regression baseline, isolated from the live service |
| `tests/` | Test suites; `tests/oracle/` holds the casbin reference used only by tests |
| `docs/` | `POLICY-CONTRACT.md`, `TECHNICAL-REQUIREMENTS.md`, `TEST-REPORT.md`, `diagrams/` |
| `STATE.md`, `REPORT.md`, `plans/`, `review/` | Working records from the build |

## Policy layer

The raw Python layer is the only runtime decision maker. A decision is over
(action, resource, context); rules are one of eight kinds (`min_metric`, `max_metric`,
`flag_true`, `flag_false`, `status_equals`, `budget_ok`, `role_in`, `no_nan`) and may be
scoped to an `action` and `resource` with casbin `keyMatch` patterns (`*`, `prefix*`).
Every applicable block rule must pass, warn rules never block, an empty bundle denies, and
anything malformed fails closed. Loosening a policy needs `human_approved_by`.
Full contract: [docs/POLICY-CONTRACT.md](docs/POLICY-CONTRACT.md).

casbin is not part of the server. It is a test-time oracle: `tests/test_policy_vs_casbin_oracle.py`
re-expresses the contract in casbin's matcher language and requires the raw layer to agree
on 10,800 seeded random decisions (3 seeds x 3,600) and on each rule's verdict, and shows the comparison detects
three deliberately broken engines.

Not yet done: policy does not gate model transitions in the running service (the registry is
built without a decision point). See the contract's Enforcement section for the planned,
default-off setting.

## Status and known gaps

- Framework-share gate: 36.5% measured (`python3 log/framework_share.py`) against a 45% target; `STATE.md` records the gate as blocking release.
- The drift worker re-checks a fixed window; there is no ingestion endpoint yet.
- The lineage graph is in memory; the MLflow mirror swallows errors by design (best effort).
- `docs/TEST-REPORT.md` and `REPORT.md` predate the connect-pipeline and policy work and are not regenerated.
- Diagrams: [docs/diagrams/](docs/diagrams/) (12 boards, source files plus a README index).
