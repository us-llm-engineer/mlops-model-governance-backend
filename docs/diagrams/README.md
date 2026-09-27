# System design diagrams

Twelve boards, one file each (`*.dc.html`), laid out by `canvas.json`. They were authored as a
design canvas; these are the sources. The `support.js` each page loads is supplied by that
canvas runtime and is not stored here, so open them through the canvas, not directly.

| # | File | Shows |
|---|---|---|
| 1 | Main | System architecture |
| 2 | RequestLifecycle | What happens to `POST /v1/models` |
| 3 | WiringArchitecture | `python -m mlops.svc`, step by step |
| 4 | PackageLayout | Which modules are live, legacy, simulation, not yet wired |
| 5 | ModelLifecycle | A model version, from register to archive |
| 6 | PolicyDecisionFlow | Policy publish/activate/decide, and the casbin test oracle |
| 7 | DriftPipeline | From a one-shot check to a scheduled sweep |
| 8 | IncidentLifecycle | Open, work the checklist, escalate, resolve |
| 9 | AuditChain | Every audit entry signs the one before it |
| 10 | LineageGraph | A lineage graph that grows with every model change |
| 11 | MlflowMirror | An opt-in shadow copy, never the source of truth |
| 12 | ConnectPipelineBeforeAfter | What the connect-pipeline pass changed |

Last synced with the code on 2026-09-27 (boards 1, 4, 6 and 12 updated for casbin becoming a
test-only oracle and for rule scope). Not yet viewed rendered in a browser.
