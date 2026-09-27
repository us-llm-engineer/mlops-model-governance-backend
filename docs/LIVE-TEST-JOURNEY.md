# Live-test journey

An end-to-end test approach for the governance backend above `exec/mlops/`: three scenarios, six
real datasets, one real `python -m mlops.svc` subprocess driven entirely through its HTTP API (plus
one CLI subprocess call and one in-process SDK call, exercising the SDK's own intended path) — no
mocks, nothing reached behind the interface. Research grounding for every non-obvious design
decision below is in [`RESEARCH-NOTES.md`](RESEARCH-NOTES.md).

## Six real datasets, three scenarios

| Scenario | Datasets | What it exercises |
|---|---|---|
| S1 — payments risk desk | ULB Credit Card Fraud (284,807 rows, 0.17% positive), AG News (127,600 articles) | RBAC, dataset/model registries, idempotency, policy gating, drift check, alert-fatigue attack, audit tamper detection, durability under `kill -9`, MLflow mirror, SBOM, calibration CLI |
| S2 — city demand forecasting | NYC TLC Yellow Taxi 2019 (stratified-sampled to 12.7M rows), Jena Climate 2009-2016 (420,551 rows) | Regression gating on real MAE, two independent named drift monitors, promote/rollback durability |
| S3 — land-cover monitoring | UCI Covertype (581,012 rows, 7 classes), EuroSAT RGB (16,200 image tiles, 10 classes) | Elevation-ordered split as a genuine distribution shift, periodic drift-evaluation worker, image-pipeline lineage |

Every trainer uses a time-forward (not random) split — see `RESEARCH-NOTES.md`'s TabReD entry —
and every regression/classification metric is the real held-out score the policy layer gates on,
never a training-set number.

## Four defect classes, found once and closed everywhere they recurred

The same four root causes surfaced independently across the three scenarios; each was fixed once
and confirmed closed on every recurrence with a real, re-measured demonstration:

1. **The drift-check endpoint crashed (HTTP 500) on any real, ordinary shift.** Root cause: the raw
   PSI computation binned a shifted window against reference-only quantile edges, and `numpy`'s
   histogram silently drops out-of-range values rather than clamping them — when a window shifts
   enough that none of its values fall in the reference's range, the proportions divide 0/0 to NaN,
   and the route serialized that NaN unguarded. Reproduced independently on fraud (V14), Jena
   (temperature), and Covertype (elevation) before being fixed once.
2. **A policy-denied model could reach production by lying about its own metric.** The gate
   correctly denied a genuinely underperforming classifier's honest metric, then approved the
   identical model when the same request claimed a fabricated passing score — metrics were
   self-attested with nothing checking them against the artifact. Closed by recording the evidence
   at validation time, tied to the artifact hash, and having the policy layer read that recorded
   evidence rather than the caller's claim. Re-demonstrated as closed on three different models,
   across both `min_metric` and `max_metric` policy-rule kinds.
3. **The durable store could silently disagree with in-memory state after a promotion/rollback.**
   A stale second "production" row survived a rollback in the SQL store, found on the city-demand
   scenario — the most severe single finding across all three runs. Fixed for every scenario.
4. **A worker-persisted value could exist with no route to ever retrieve it.** The periodic
   drift-evaluation worker kept computing and saving a real baseline behind the crashing endpoint's
   back, so a caller was blind twice over: a 500 with no information, and no way to fetch what had
   quietly been saved anyway. Closed with an additive `GET` route once the underlying crash was
   fixed.

## Statistics-collection pass

A second pass over the same six trained models adds real post-training evidence beyond a single
scalar metric: confusion matrices and per-class precision/recall/F1, a calibration curve with ECE
and Brier score for the fraud classifier, permutation feature importance for the four tabular
models, three-quantile prediction bands for the two regressors, per-class token saliency for the
text classifier, and a diagnostic training curve for every model. See the README's results table
and figure grid, and `RESEARCH-NOTES.md` for what motivated each addition and — just as
importantly — what each one explicitly does *not* claim to be.

## Current status

All three scenarios are complete and re-verified against the fixed code. Two items are explicitly
open rather than silently dropped: an alert-fatigue experiment whose first attempt was confounded
by the fraud dataset's real non-stationarity (the "driftless" baseline turned out not to be
driftless — an accurate finding about the data, not a bug, but one that means the experiment needs
a different baseline-construction design to actually demonstrate cry-wolf behavior); and a planned
mutation-testing pass across all three scenarios' now-passing flows, not yet run.
